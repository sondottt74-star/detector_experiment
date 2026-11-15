"""把复现包整理成一个只保留实验输入的 ModelScope 数据集。"""

import argparse
import json
import shutil
from pathlib import Path


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


def rewrite_row(row):
    chinese_name = row["original_can_type"]
    can_name = PINYIN[chinese_name]
    capture_id = Path(row["image"]).parent.name
    image_name = Path(row["image"]).name
    defect_name = Path(row["defect_mask"]).name if row["defect_mask"] else None
    foreground_name = Path(row["foreground_mask"]).name
    annotation_name = Path(row["source_annotation"]).name
    return {
        **row,
        "can_type": can_name,
        "original_can_type": can_name,
        "can_name_zh": chinese_name,
        "model_id": row["can_type"],
        "image": f"images/{can_name}/{capture_id}/{image_name}",
        "foreground_mask": f"masks/{can_name}/{capture_id}/{foreground_name}",
        "defect_mask": f"masks/{can_name}/{capture_id}/{defect_name}" if defect_name else None,
        "source_annotation": f"annotations/{can_name}/{capture_id}/{annotation_name}",
    }


def prepare(source, output):
    source = Path(source)
    output = Path(output)
    data = source / "files/data"
    installed = source / "installed/dataset_s43_20261020"
    output.mkdir(parents=True, exist_ok=True)

    for directory in ("images", "masks", "annotations"):
        for chinese_name, can_name in PINYIN.items():
            shutil.copytree(
                data / "processed" / directory / chinese_name,
                output / directory / can_name,
            )

    mapping = json.loads((data / "processed/mapping.json").read_text(encoding="utf-8"))
    metadata = {
        can_name: {"name_zh": chinese_name, "model_id": model_id}
        for chinese_name, model_id in mapping.items()
        for can_name in [PINYIN[chinese_name]]
    }
    (output / "metadata").mkdir(exist_ok=True)
    (output / "metadata/name_map.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    for chinese_name, can_name in PINYIN.items():
        model_id = mapping[chinese_name]
        shutil.copytree(
            data / "references" / model_id,
            output / "references" / can_name,
        )

    bank = json.loads((data / "processed/reference_bank.json").read_text(encoding="utf-8"))
    clean_bank = {}
    for chinese_name, model_id in mapping.items():
        can_name = PINYIN[chinese_name]
        clean_bank[can_name] = [
            {
                **record,
                "image": f"references/{can_name}/{Path(record['image']).name}",
                "foreground_mask": f"references/{can_name}/{Path(record['foreground_mask']).name}",
            }
            for record in bank[model_id]
        ]
    (output / "metadata/reference_bank.json").write_text(
        json.dumps(clean_bank, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    splits = output / "splits"
    splits.mkdir(exist_ok=True)
    for fold in range(1, 6):
        rows = [
            rewrite_row(json.loads(line))
            for line in (installed / f"fold{fold}/dataset_indexed.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        (splits / f"fold{fold}.jsonl").write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
            encoding="utf-8",
        )

    (output / "README.md").write_text(
        """# can_detect clean dataset

目录只保留检测实验使用的图像、掩码、标注、3DGS 参考图库和五折清单。

- `images/<pinyin>/<capture>/`: 待测图
- `masks/<pinyin>/<capture>/`: foreground.png 与 defect.png
- `annotations/<pinyin>/<capture>/`: 原始标注 JSON
- `references/<pinyin>/`: 3DGS 正常参考图与参考前景掩码
- `splits/fold1.jsonl` 到 `fold5.jsonl`: 五折实验清单
- `metadata/name_map.json`: 拼音名、中文名和 3DGS 模型 ID
- `metadata/reference_bank.json`: 参考图库索引

清单中的路径都相对于数据集根目录。
""",
        encoding="utf-8",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    prepare(args.source, args.output)
