"""整理 3DGS 建库输入和全部待测图原始输入。"""

import argparse
import json
import shutil
from pathlib import Path


# 中文品牌名改成目录用的拼音，避免路径里出现中文。
PINYIN = {
    "冰峰": "bingfeng",
    "哈尔滨": "haerbin",
    "哈尔滨清爽": "haerbin_qingshuang",
    "红牛": "hongniu",
    "嘉士伯": "jiashibo",
    "凯撒": "kaisa",
    "凯撒黑": "kaisa_hei",
    "科罗娜啤酒": "keluona_pijiu",
    "可乐": "kele",
    "酸梅汤": "suanmeitang",
    "无糖百事": "wutang_baishi",
    "rio清爽": "rio_qingshuang",
    "riolight": "rio_light",
}


def source_row(row):
    """把原始索引里的中文路径改写成整理后的相对路径。"""
    chinese_name = row["original_can_type"]
    can_name = PINYIN[chinese_name]
    capture_id = Path(row["image"]).parent.name
    image_name = Path(row["image"]).name
    annotation_name = Path(row["source_annotation"]).name
    image = f"images/{can_name}/{capture_id}/{image_name}"
    annotation = f"annotations/{can_name}/{capture_id}/{annotation_name}"
    return {
        "image": image,
        "annotation": annotation,
        "can_type": can_name,
        "can_name_zh": chinese_name,
        "model_id": row["can_type"],
        "capture_id": row["capture_id"],
        "physical_can_id": row["physical_can_id"],
        "origin": row["origin"],
        "source_frame": row.get("source_frame"),
        "label": row["label"],
        "split": row["split"],
        # source_frame 以 frame_ 开头的是建库视频抽帧，其余是单独拍摄的待测图。
        "is_3dgs_input": str(row.get("source_frame", "")).startswith("frame_"),
    }


def rewrite_annotation(source, target, can_name, capture_id, image_name):
    """改写 LabelMe 的 imagePath，让标注指向整理后的图像位置。"""
    annotation = json.loads(source.read_text(encoding="utf-8"))
    annotation["imagePath"] = f"../../../images/{can_name}/{capture_id}/{image_name}"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(annotation, ensure_ascii=False, indent=2), encoding="utf-8")


def prepare(source, output):
    source = Path(source)
    output = Path(output)
    data = source / "files/data"
    installed = source / "installed/dataset_s43_20261020"
    images_root = output / "images"
    annotations_root = output / "annotations"
    images_root.mkdir(parents=True, exist_ok=True)
    annotations_root.mkdir(parents=True, exist_ok=True)

    # 记下拼音名、中文名和 3DGS 模型 ID 的对应关系。
    mapping = json.loads((data / "processed/mapping.json").read_text(encoding="utf-8"))
    metadata = {
        PINYIN[chinese_name]: {"name_zh": chinese_name, "model_id": model_id}
        for chinese_name, model_id in mapping.items()
    }
    (output / "metadata").mkdir(exist_ok=True)
    (output / "metadata/name_map.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 图像按品牌和拍摄批次原样复制；标注里的 imagePath 改到新目录。
    for chinese_name, can_name in PINYIN.items():
        shutil.copytree(
            data / "processed/images" / chinese_name,
            images_root / can_name,
        )
        for source_annotation in (data / "processed/annotations" / chinese_name).rglob("*.json"):
            capture_id = source_annotation.parent.name
            image_name = source_annotation.stem + ".jpg"
            target_annotation = annotations_root / can_name / capture_id / source_annotation.name
            rewrite_annotation(
                source_annotation,
                target_annotation,
                can_name,
                capture_id,
                image_name,
            )

    # fold1 同时作为全部待测图清单；其中抽帧记录单独写成 3DGS 建库输入。
    rows = [
        source_row(json.loads(line))
        for line in (installed / "fold1/dataset_indexed.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    manifests = output / "manifests"
    manifests.mkdir(exist_ok=True)
    (manifests / "query_inputs.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    reference_rows = [row for row in rows if row["is_3dgs_input"]]
    (manifests / "reference_inputs.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in reference_rows) + "\n",
        encoding="utf-8",
    )

    # 五折清单只改路径，划分本身沿用交付包。
    for fold in range(1, 6):
        fold_rows = [
            source_row(json.loads(line))
            for line in (installed / f"fold{fold}/dataset_indexed.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        (manifests / f"fold{fold}.jsonl").write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in fold_rows) + "\n",
            encoding="utf-8",
        )

    (output / "README.md").write_text(
        """# can_detect raw inputs

这里是实验的原始输入重构目录，不包含已经建立好的 3DGS 参考库，也不包含已经生成的前景掩码、缺陷掩码或 `reference_bank.json`。

- `images/<pinyin>/<capture>/`: 全部 1625 张原始图像，包含正常图和异常图
- `annotations/<pinyin>/<capture>/`: 与原始图像对应的 LabelMe 标注
- `manifests/reference_inputs.jsonl`: 建立 3DGS 参考库的原始正常视频帧输入
- `manifests/query_inputs.jsonl`: 全部可以作为待测图输入的原始图像
- `manifests/fold1.jsonl` 到 `fold5.jsonl`: 五折原始输入清单
- `metadata/name_map.json`: 拼音名、中文名和 3DGS 模型 ID

`is_3dgs_input=true` 的记录来自原始建库视频抽帧；其余正常照片和异常照片都保留在 `query_inputs.jsonl` 中供待测流程使用。

交付包本身没有保存原始 mp4，只有从建库视频得到的原始 jpg 帧；因此 3DGS 建库可直接使用 `reference_inputs.jsonl`，也可以替换成用户手里的原始视频后重新抽帧。
""",
        encoding="utf-8",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    prepare(args.source, args.output)
