"""待测图预处理：SAM2、518 正方形、按 tag 检索参考图，写出训练和验证清单。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image
import torch
import yaml

from detector.utils import fit_foreground, load_local_dinov2
from preprocessing.build_detection_manifest import encode_paths, write_target_mask
from preprocessing.segment_foreground import _config_name


ROOT = Path(__file__).resolve().parents[1]
# 与 segment_foreground.py 相同：框是相对宽高，三个正点在罐体中线上。
PROMPT_BOX = (0.2, 0.1, 0.8, 0.95)
# 六折共用同一份 518 图。键是原图绝对路径，不按折再存一份。
SHARED = ROOT / "prepared" / "_shared"
QUERY_CACHE = SHARED / "queries"
REFERENCE_CACHE = SHARED / "references"
# 上一轮只缓存了原始 SAM 掩码。518 图还没有时，用它免掉再次分割。
LEGACY_SAM_CACHE = ROOT / "prepared" / "_sam_cache"
# 检索编码和训练 batch 分开。训练仍用 yaml 里的 batch_size。
ENCODE_BATCH = 32


def resolve(path):
    # 配置里的相对路径相对 detector_experiment 目录。
    if path is None:
        return None
    path = Path(path)
    if path.is_absolute():
        return path
    return ROOT / path


def locate(root, value):
    # fold 清单里的图片和标注路径相对工作区根目录。
    path = Path(value)
    if path.is_absolute():
        return path
    return root / path


def _xyxy(box):
    x0, y0, x1, y1 = [float(value) for value in box]
    return [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]


def defect_boxes(annotation_path, image_path=None):
    """LabelMe 用两个角点；新测试数据是整目录一份 xyxy 清单，按文件名对齐。"""
    payload = json.loads(Path(annotation_path).read_text(encoding="utf-8"))
    if "shapes" in payload:
        boxes = []
        for shape in payload["shapes"]:
            # can 是整罐外框。缺陷框在旧标注里叫 dent，在另一部分标注里叫 defect。
            if shape["shape_type"] != "rectangle" or shape.get("label") not in ("defect", "dent"):
                continue
            (x0, y0), (x1, y1) = shape["points"]
            boxes.append(_xyxy((x0, y0, x1, y1)))
        return boxes
    if "records" in payload:
        image_name = Path(image_path).name if image_path else None
        matched = [
            rec for rec in payload["records"]
            if image_name is None or rec.get("file") == image_name or Path(rec.get("image") or "").name == image_name
        ]
        if image_name is not None and not matched:
            raise FileNotFoundError(f"no box record for {image_name} in {annotation_path}")
        return [_xyxy(box) for rec in matched for box in (rec.get("boxes") or [])]
    raise KeyError(f"unsupported annotation: {annotation_path}")


def can_box(annotation_path):
    """LabelMe 的整罐框。新测试数据没有这个框，返回 None，继续用固定提示。"""
    payload = json.loads(Path(annotation_path).read_text(encoding="utf-8"))
    if "shapes" not in payload:
        return None
    boxes = []
    for shape in payload["shapes"]:
        if shape.get("shape_type") != "rectangle" or shape.get("label") != "can":
            continue
        (x0, y0), (x1, y1) = shape["points"]
        boxes.append(_xyxy((x0, y0, x1, y1)))
    if not boxes:
        return None
    return max(boxes, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]))


def load_bank(path):
    # 有的参考库按品牌嵌在 brands 里，当前这份直接用模型编号当键。
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if "brands" in payload:
        return payload["brands"]
    return payload


def segment_image(predictor, image, box=None):
    """有整罐框时用该框，否则用相对画面的固定框。三个正点都在框内中线上。"""
    array = np.asarray(image)
    height, width = array.shape[:2]
    if box is None:
        image_box = np.asarray(
            [
                PROMPT_BOX[0] * width,
                PROMPT_BOX[1] * height,
                PROMPT_BOX[2] * width,
                PROMPT_BOX[3] * height,
            ],
            dtype=np.float32,
        )
    else:
        x0, y0, x1, y1 = box
        image_box = np.asarray(
            [
                min(max(x0, 0), width - 1),
                min(max(y0, 0), height - 1),
                min(max(x1, 0), width - 1),
                min(max(y1, 0), height - 1),
            ],
            dtype=np.float32,
        )
    points = np.asarray(
        [
            [(image_box[0] + image_box[2]) / 2, image_box[1] + ratio * (image_box[3] - image_box[1])]
            for ratio in (0.3, 0.5, 0.7)
        ],
        dtype=np.float32,
    )
    predictor.set_image(array)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        masks, scores, _ = predictor.predict(
            box=image_box,
            point_coords=points,
            point_labels=np.ones(3, dtype=np.int32),
            multimask_output=True,
        )
    mask = masks[int(scores.argmax())]
    return Image.fromarray(mask.astype(np.uint8) * 255)


def cache_key(path):
    return hashlib.sha1(str(Path(path).resolve()).encode()).hexdigest()


def _nonempty_file(path):
    path = Path(path)
    return path.is_file() and path.stat().st_size > 0


def query_cache_paths(image_path):
    dest = QUERY_CACHE / cache_key(image_path)
    return {
        "sam": dest / "sam.png",
        "query": dest / "query.jpg",
        "mask": dest / "query_mask.png",
        "target": dest / "target.png",
    }


def reference_cache_paths(image_path):
    dest = REFERENCE_CACHE / cache_key(image_path)
    return dest / "ref.jpg", dest / "ref_mask.png"


def cached_sam_mask(image_path):
    """已有的 SAM 掩码直接读，不再跑 SAM2。"""
    paths = query_cache_paths(image_path)
    if _nonempty_file(paths["sam"]):
        with Image.open(paths["sam"]) as cached:
            return cached.copy()
    legacy = LEGACY_SAM_CACHE / f"{cache_key(image_path)}.png"
    if _nonempty_file(legacy):
        paths["sam"].parent.mkdir(parents=True, exist_ok=True)
        with Image.open(legacy) as cached:
            mask = cached.copy()
        mask.save(paths["sam"])
        return mask
    return None


def fit_references(items, size):
    # 参考图只在共享目录里放一份。空掩码没有外接框，跳过，不让 xs.min 崩溃。
    fitted = []
    skipped = 0
    reused = 0
    fitted_now = 0
    for item in items:
        source = Path(item["image"])
        image_out, mask_out = reference_cache_paths(source)
        if _nonempty_file(image_out) and _nonempty_file(mask_out):
            fitted.append({
                "image": str(image_out.resolve()),
                "foreground_mask": str(mask_out.resolve()),
            })
            reused += 1
            continue
        with Image.open(item["foreground_mask"]) as raw_mask:
            if not (np.asarray(raw_mask.convert("L")) > 127).any():
                skipped += 1
                print(f"[prepare] skip empty mask {item['foreground_mask']}", flush=True)
                continue
            image, mask, _ = fit_foreground(Image.open(source), raw_mask, size)
        image_out.parent.mkdir(parents=True, exist_ok=True)
        image.save(image_out, quality=95)
        mask.save(mask_out)
        fitted_now += 1
        fitted.append({
            "image": str(image_out.resolve()),
            "foreground_mask": str(mask_out.resolve()),
        })
    print(
        f"[prepare] references kept={len(fitted)} reused={reused} "
        f"fitted={fitted_now} skipped_empty={skipped}",
        flush=True,
    )
    return fitted, skipped


def prepare(config_path, splits=("train", "val"), gallery_cache=None):
    with Path(config_path).open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    source_manifest = resolve(config["source_manifest"])
    # fold1/manifest.json 的上两级是工作区根目录，清单里的相对路径从这里拼。
    data_root = source_manifest.parent.parent
    prepared = resolve(config["prepared_dir"])
    size = config["image_size"]
    payload = json.loads(source_manifest.read_text(encoding="utf-8"))
    # 建库抽帧不进待测。训练和验证沿用清单里的 split。
    splits = set(splits)
    queries = [
        record for record in payload["records"]
        if record["split"] in splits and record["role"] != "reference_video_frame"
    ]
    for record in queries:
        record["tag"] = record["brand"]

    bank = load_bank(resolve(config["reference_bank"]))
    encoder = load_local_dinov2(
        resolve(config["dino_repo"]), resolve(config["dino_weights"])
    ).to(config["device"])
    predictor = None

    def ensure_predictor():
        # 缓存全命中时不加载 SAM2。一个进程里也只建一次。
        nonlocal predictor
        if predictor is None:
            sys.path.insert(0, str(resolve(config["sam2_repo"])))
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            predictor = SAM2ImagePredictor(
                build_sam2(
                    _config_name(resolve(config["sam_config"]), resolve(config["sam2_repo"])),
                    str(resolve(config["sam_checkpoint"])),
                    device=config["device"],
                )
            )
        return predictor

    rows = {split: [] for split in splits}
    by_tag = {}
    for record in queries:
        by_tag.setdefault(record["tag"], []).append(record)

    empty_skips = 0
    query_hits = 0
    query_fits = 0
    for tag, records in by_tag.items():
        # 参考库按 3DGS 模型编号分组，同一品牌的待测图共用这一组。
        model_id = records[0]["model_id"]
        gallery_items = bank[model_id]
        print(
            f"[prepare] tag={tag} fitting gallery={len(gallery_items)} queries={len(records)}",
            flush=True,
        )
        # 同一参考库的检索向量在多份配置之间复用，测试集不再重编码。
        if gallery_cache is not None and model_id in gallery_cache:
            gallery, gallery_vectors = gallery_cache[model_id]
        else:
            gallery, skipped = fit_references(gallery_items, size)
            empty_skips += skipped
            if not gallery:
                raise RuntimeError(f"no usable reference views for {tag}")
            gallery_vectors = encode_paths(
                encoder,
                [item["image"] for item in gallery],
                ENCODE_BATCH,
                config["device"],
            )
            if gallery_cache is not None:
                gallery_cache[model_id] = (gallery, gallery_vectors)
        fitted_queries = []
        for index, record in enumerate(records):
            image_path = locate(data_root, record["image"])
            cached = query_cache_paths(image_path)
            if index % 50 == 0:
                print(
                    f"[prepare] tag={tag} query {index}/{len(records)} "
                    f"hits={query_hits} fits={query_fits}",
                    flush=True,
                )
            if all(_nonempty_file(cached[name]) for name in ("sam", "query", "mask", "target")):
                query_hits += 1
                fitted_queries.append({
                    "query": str(cached["query"].resolve()),
                    "query_mask": str(cached["mask"].resolve()),
                    "target_mask": str(cached["target"].resolve()),
                    "label": int(record["label"]),
                    "tag": tag,
                    "split": record["split"],
                })
                continue
            image = Image.open(image_path).convert("RGB")
            mask = cached_sam_mask(image_path)
            if mask is None:
                mask = segment_image(ensure_predictor(), image)
                cached["sam"].parent.mkdir(parents=True, exist_ok=True)
                mask.save(cached["sam"])
            fitted, foreground, window = fit_foreground(image, mask, size)
            cached["query"].parent.mkdir(parents=True, exist_ok=True)
            fitted.save(cached["query"], quality=95)
            foreground.save(cached["mask"])
            boxes = defect_boxes(locate(data_root, record["annotation"]), image_path) if record["annotation"] else []
            write_target_mask(boxes, window, cached["target"], size)
            query_fits += 1
            fitted_queries.append({
                "query": str(cached["query"].resolve()),
                "query_mask": str(cached["mask"].resolve()),
                "target_mask": str(cached["target"].resolve()),
                "label": int(record["label"]),
                "tag": tag,
                "split": record["split"],
            })
        print(
            f"[prepare] tag={tag} queries={len(records)} cache_hits={query_hits} fits={query_fits}",
            flush=True,
        )
        query_vectors = encode_paths(
            encoder,
            [item["query"] for item in fitted_queries],
            ENCODE_BATCH,
            config["device"],
        )
        # 向量已归一化，点积就是余弦相似度。只在当前 tag 的参考库里取前若干张。
        similarity = query_vectors @ gallery_vectors.T
        count = min(config["reference_count"], len(gallery))
        for item, scores in zip(fitted_queries, similarity):
            indices = scores.topk(count).indices.tolist()
            item["references"] = [gallery[index]["image"] for index in indices]
            item["reference_masks"] = [gallery[index]["foreground_mask"] for index in indices]
            split = item.pop("split")
            rows[split].append(item)
    # 只写出这次指定的划分，避免测试流程重写已经完成的训练和验证清单。
    written = []
    for split, items in rows.items():
        path = prepared / f"{split}.jsonl"
        write_rows(path, items)
        written.append(f"{path} n={len(items)}")
    print(
        f"[prepare] wrote {' '.join(written)} "
        f"empty_mask_skips={empty_skips} query_hits={query_hits} query_fits={query_fits}",
        flush=True,
    )


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="+")
    parser.add_argument("--splits", default="train,val")
    args = parser.parse_args()
    splits = tuple(split.strip() for split in args.splits.split(",") if split.strip())
    gallery_cache = {}
    for config_path in args.config:
        prepare(config_path, splits, gallery_cache)


if __name__ == "__main__":
    main()
