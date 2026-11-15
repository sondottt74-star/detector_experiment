"""Create detector manifests from SAM2 query masks and a rendered reference bank.

Reference selection is a DINOv2 cosine retrieval step. The selected normal
reference images are then consumed by model.py's cross-attention detector.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

from detector.utils import load_local_dinov2, load_rgb


def write_target_mask(boxes, width, height, path, size):
    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)
    for box in boxes:
        if len(box) != 4:
            continue
        x0, y0, x1, y1 = [float(v) for v in box]
        scaled = (
            round(max(0, min(size, x0 / width * size))),
            round(max(0, min(size, y0 / height * size))),
            round(max(0, min(size, x1 / width * size))),
            round(max(0, min(size, y1 / height * size))),
        )
        draw.rectangle(scaled, fill=255)
    path.parent.mkdir(parents=True, exist_ok=True)
    mask.save(path)


def fixed_mask(path, size):
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rectangle(
        (int(.20 * size), int(.10 * size), int(.80 * size), int(.95 * size)), fill=255
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    mask.save(path)


def encode_paths(encoder, paths, image_size, batch_size, device):
    vectors = []
    encoder.eval()
    with torch.inference_mode():
        for start in range(0, len(paths), batch_size):
            images = torch.stack(
                [load_rgb(path, image_size) for path in paths[start:start + batch_size]]
            ).to(device)
            tokens = encoder(pixel_values=images).last_hidden_state
            vector = F.normalize(tokens[:, 1:].mean(dim=1), dim=-1)
            vectors.append(vector.cpu())
    return torch.cat(vectors, dim=0)


def load_bank(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload["brands"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--reference-bank", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dino-repo", required=True)
    parser.add_argument("--dino-weights", default=None)
    parser.add_argument("--sam2-mask-root", required=True)
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--reference-count", type=int, default=4)
    parser.add_argument("--retrieval-batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    output = Path(args.output).resolve()
    target_root = output / "masks"
    test_root = data_root / "新测试数据"
    bank = load_bank(args.reference_bank)
    device = torch.device(args.device)
    encoder = load_local_dinov2(args.dino_repo, args.dino_weights).to(device)

    rows_by_brand = {}
    for brand_dir in sorted(p for p in test_root.iterdir() if p.is_dir()):
        payload = json.loads((brand_dir / "annotations.json").read_text(encoding="utf-8"))
        records = []
        for rec in sorted(payload["records"], key=lambda x: (x["can_id"], x["file"])):
            image_path = brand_dir / "images" / rec["file"]
            stem = Path(rec["file"]).stem
            query_mask = Path(args.sam2_mask_root) / brand_dir.name / f"{stem}_mask.png"
            if not query_mask.exists():
                raise FileNotFoundError(f"SAM2 mask missing: {query_mask}")
            target_mask = target_root / brand_dir.name / f"{stem}_target.png"
            write_target_mask(rec.get("boxes", []), int(rec["width"]), int(rec["height"]), target_mask, args.image_size)
            records.append({
                "brand": brand_dir.name,
                "can_id": str(rec["can_id"]),
                "query": str(image_path.resolve()),
                "query_mask": str(query_mask.resolve()),
                "target_mask": str(target_mask.resolve()),
                "label": int(rec["label_id"]),
            })
        rows_by_brand[brand_dir.name] = records

    retrieval_cache = output / "retrieval_cache"
    retrieval_cache.mkdir(parents=True, exist_ok=True)
    for brand, rows in sorted(rows_by_brand.items()):
        if brand not in bank:
            aliases = [name for name in bank if brand in name or name in brand]
            if len(aliases) != 1:
                raise KeyError(f"reference bank has no unambiguous entry for {brand}")
            bank_entry = bank[aliases[0]]
        else:
            bank_entry = bank[brand]
        gallery = bank_entry["references"]
        gallery_paths = [item["image"] for item in gallery]
        cache_path = retrieval_cache / f"{brand}.pt"
        if cache_path.exists():
            gallery_vectors = torch.load(cache_path, map_location="cpu", weights_only=True)
        else:
            gallery_vectors = encode_paths(
                encoder, gallery_paths, args.image_size, args.retrieval_batch_size, device
            )
            torch.save(gallery_vectors, cache_path)
        query_vectors = encode_paths(
            encoder, [row["query"] for row in rows], args.image_size,
            args.retrieval_batch_size, device
        )
        similarity = query_vectors @ gallery_vectors.T
        for row, scores in zip(rows, similarity):
            count = min(args.reference_count, len(gallery))
            indices = scores.topk(count).indices.tolist()
            row["references"] = [gallery[index]["image"] for index in indices]
            row["reference_masks"] = [gallery[index]["foreground_mask"] for index in indices]
            row["reference_scores"] = [float(scores[index]) for index in indices]

    all_rows = [row for rows in rows_by_brand.values() for row in rows]
    train, val = [], []
    for brand, rows in sorted(rows_by_brand.items()):
        groups = sorted({row["can_id"] for row in rows})
        anomaly_count = {group: sum(row["label"] == 1 for row in rows if row["can_id"] == group) for group in groups}
        val_groups = set(sorted(groups, key=lambda group: (-anomaly_count[group], group))[:max(1, int(len(groups) * .3))])
        train.extend(row for row in rows if row["can_id"] not in val_groups)
        val.extend(row for row in rows if row["can_id"] in val_groups)

    output.mkdir(parents=True, exist_ok=True)
    for name, records in (("train", train), ("val", val), ("all", all_rows)):
        (output / f"{name}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8"
        )
    summary = {
        "total": len(all_rows), "train": len(train), "val": len(val),
        "anomalies": sum(row["label"] for row in all_rows),
        "reference_count": args.reference_count,
        "matching": "DINOv2 cosine similarity over gallery mean patch tokens",
        "query_foreground": "SAM2 masks",
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
