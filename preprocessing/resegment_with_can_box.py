"""有整罐框的待测图用该框重新提示 SAM，并按新图重选参考图。

没有整罐框的图不改分割，也不改参考列表。
"""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from detector.utils import fit_foreground, load_local_dinov2
from preprocessing.build_detection_manifest import encode_paths, write_target_mask
from preprocessing.prepare_queries import cache_key, can_box, defect_boxes, load_bank, segment_image
from preprocessing.segment_foreground import _config_name

DATA_ROOT = Path("/home/zhanglesong")
MANIFESTS = [DATA_ROOT / f"fold{fold}/manifest.json" for fold in range(1, 6)]
MANIFESTS.append(DATA_ROOT / "zeroshot/manifest.json")
QUERY_CACHE = ROOT / "cleaned_data" / "_shared" / "queries"
REFERENCE_CACHE = ROOT / "cleaned_data" / "_shared" / "references"
CLEANED = ROOT / "cleaned_data"
SIZE = 518
REFERENCE_COUNT = 4
SAM_REPO = ROOT / "third_party" / "sam2"
SAM_CHECKPOINT = Path("/home/zhanglesong/can_anomaly_detection/models/sam2.1_hiera_b_plus/sam2.1_hiera_base_plus.pt")
SAM_CONFIG = SAM_REPO / "sam2" / "configs" / "sam2.1" / "sam2.1_hiera_b+.yaml"
DINO_REPO = Path("/home/zhanglesong/can_anomaly_detection/models/dinov2_vits14")
BANK_PATH = Path("/home/zhanglesong/can_anomaly_detection/data/reference/reference_bank.json")
DEVICE = "cuda:0"


def locate(value):
    path = Path(value)
    if path.is_absolute():
        return path
    return DATA_ROOT / path


def query_records():
    seen = set()
    for manifest in MANIFESTS:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        for record in payload["records"]:
            if record.get("role") == "reference_video_frame":
                continue
            image_path = locate(record["image"]).resolve()
            if image_path in seen:
                continue
            seen.add(image_path)
            yield record, image_path


def local_query_copies():
    """fold2、fold3 清单指向这里的副本，文件名里带原图 stem。"""
    found = {}
    for fold in (2, 3):
        for target_path in (CLEANED / f"fold{fold}" / "queries").rglob("*_target.png"):
            prefix = target_path.name[: -len("_target.png")]
            stem = prefix.split("_", 1)[1]
            brand = target_path.parent.name
            found.setdefault((brand, stem), []).append({
                "query": target_path.with_name(prefix + ".jpg"),
                "mask": target_path.with_name(prefix + "_mask.png"),
                "sam": target_path.with_name(prefix + "_sam.png"),
                "target": target_path,
            })
    return found


def load_predictor():
    sys.path.insert(0, str(SAM_REPO))
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    return SAM2ImagePredictor(
        build_sam2(
            _config_name(SAM_CONFIG, SAM_REPO),
            str(SAM_CHECKPOINT),
            device=DEVICE,
        )
    )


def resegment(predictor, copies):
    updated = {}
    skipped_empty = 0
    for index, (record, image_path) in enumerate(query_records(), start=1):
        annotation = record.get("annotation")
        box = can_box(locate(annotation)) if annotation else None
        if box is None:
            continue
        with Image.open(image_path) as image:
            mask = segment_image(predictor, image.convert("RGB"), box)
            if not np.asarray(mask).any():
                skipped_empty += 1
                print(f"[resegment] empty mask {image_path}", flush=True)
                continue
            fitted, foreground, window = fit_foreground(image, mask, SIZE)
        cached = QUERY_CACHE / cache_key(image_path)
        cached.mkdir(parents=True, exist_ok=True)
        mask.save(cached / "sam.png")
        fitted.save(cached / "query.jpg", quality=95)
        foreground.save(cached / "query_mask.png")
        boxes = defect_boxes(locate(annotation), image_path) if annotation else []
        write_target_mask(boxes, window, cached / "target.png", SIZE)
        for copy in copies.get((record.get("brand"), image_path.stem), []):
            mask.save(copy["sam"])
            fitted.save(copy["query"], quality=95)
            foreground.save(copy["mask"])
            write_target_mask(boxes, window, copy["target"], SIZE)
        updated[image_path] = record
        if len(updated) % 50 == 0:
            print(f"[resegment] updated {len(updated)} scanned {index}", flush=True)
    print(f"[resegment] updated {len(updated)} empty {skipped_empty}", flush=True)
    return updated


def reference_galleries():
    """共享缓存里每个 3DGS 模型已经做好的参考图。"""
    bank = load_bank(BANK_PATH)
    galleries = {}
    for model_id, items in bank.items():
        gallery = []
        for item in items:
            source = Path(item["image"])
            cached = REFERENCE_CACHE / cache_key(source)
            image = cached / "ref.jpg"
            mask = cached / "ref_mask.png"
            if not image.is_file() or not mask.is_file():
                continue
            gallery.append({
                "image": str(image.resolve()),
                "foreground_mask": str(mask.resolve()),
                "name": source.name,
                "brand": source.parent.name,
            })
        galleries[model_id] = gallery
        print(f"[gallery] {model_id} {len(gallery)}", flush=True)
    return galleries


def choose_references(updated, galleries):
    encoder = load_local_dinov2(DINO_REPO, None).to(DEVICE)
    vectors = {}
    for model_id, gallery in galleries.items():
        if not gallery:
            continue
        vectors[model_id] = encode_paths(
            encoder, [item["image"] for item in gallery], 32, DEVICE
        )
    chosen = {}
    by_model = {}
    for image_path, record in updated.items():
        by_model.setdefault(record["model_id"], []).append(image_path)
    for model_id, paths in by_model.items():
        gallery = galleries[model_id]
        query_vectors = encode_paths(
            encoder,
            [str(QUERY_CACHE / cache_key(path) / "query.jpg") for path in paths],
            32,
            DEVICE,
        )
        count = min(REFERENCE_COUNT, len(gallery))
        similarity = query_vectors @ vectors[model_id].T
        for image_path, scores in zip(paths, similarity):
            indices = scores.topk(count).indices.tolist()
            chosen[image_path] = [gallery[index] for index in indices]
    print(f"[retrieve] queries {len(chosen)}", flush=True)
    return chosen


def local_reference(fold, item):
    image = CLEANED / f"fold{fold}" / "references" / item["brand"] / item["name"]
    mask = image.with_name(f"{image.stem}_mask.png")
    if image.is_file() and mask.is_file():
        return str(image.resolve()), str(mask.resolve())
    return item["image"], item["foreground_mask"]


def rewrite_manifests(chosen, sources):
    hash_to_path = {cache_key(path): path for path in chosen}
    written = 0
    for name in [f"fold{fold}" for fold in range(1, 6)] + ["zeroshot"]:
        for split in ("train", "val", "test"):
            path = CLEANED / name / f"{split}.jsonl"
            if not path.is_file():
                continue
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            changed = 0
            fold = int(name[4:]) if name.startswith("fold") else None
            for row in rows:
                query = Path(row["query"])
                if "/_shared/queries/" in row["query"]:
                    image_path = hash_to_path.get(query.parent.name)
                elif fold in (2, 3):
                    stem = query.stem.split("_", 1)[1]
                    image_path = sources.get((query.parent.name, stem))
                if image_path not in chosen:
                    continue
                picks = chosen[image_path]
                if fold in (2, 3) and "/_shared/" not in row["references"][0]:
                    pairs = [local_reference(fold, item) for item in picks]
                    row["references"] = [pair[0] for pair in pairs]
                    row["reference_masks"] = [pair[1] for pair in pairs]
                else:
                    row["references"] = [item["image"] for item in picks]
                    row["reference_masks"] = [item["foreground_mask"] for item in picks]
                changed += 1
            path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )
            written += changed
            print(f"[jsonl] {name}/{split} updated {changed}/{len(rows)}", flush=True)
    print(f"[jsonl] updated rows {written}", flush=True)


def main():
    copies = local_query_copies()
    sources = {}
    for record, image_path in query_records():
        sources[(record.get("brand"), image_path.stem)] = image_path
    predictor = load_predictor()
    updated = resegment(predictor, copies)
    del predictor
    torch.cuda.empty_cache()
    chosen = choose_references(updated, reference_galleries())
    rewrite_manifests(chosen, sources)


if __name__ == "__main__":
    main()
