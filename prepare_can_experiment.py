"""Build manifests and 448px masks from the ModelScope can input dataset."""
import argparse
import json
from pathlib import Path
from PIL import Image, ImageDraw


def mask_image(boxes, width, height, out_path, foreground=False):
    canvas = Image.new("L", (448, 448), 0)
    draw = ImageDraw.Draw(canvas)
    if foreground:
        draw.rectangle((int(.20 * 448), int(.10 * 448), int(.80 * 448), int(.95 * 448)), fill=255)
    else:
        for box in boxes:
            if len(box) != 4:
                continue
            x0, y0, x1, y1 = box
            x0 = max(0, min(448, round(float(x0) / width * 448)))
            y0 = max(0, min(448, round(float(y0) / height * 448)))
            x1 = max(0, min(448, round(float(x1) / width * 448)))
            y1 = max(0, min(448, round(float(y1) / height * 448)))
            draw.rectangle((min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)), fill=255)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    data_root = Path(args.data_root)
    output = Path(args.output)
    test_root = data_root / "新测试数据"
    mask_root = output / "masks"
    rows = []
    for brand_dir in sorted(p for p in test_root.iterdir() if p.is_dir()):
        annotation_path = brand_dir / "annotations.json"
        payload = json.loads(annotation_path.read_text(encoding="utf-8"))
        for rec in sorted(payload["records"], key=lambda x: (x["can_id"], x["file"])):
            image_path = brand_dir / "images" / rec["file"]
            if not image_path.exists():
                raise FileNotFoundError(image_path)
            stem = Path(rec["file"]).stem
            safe_brand = brand_dir.name
            query_mask = mask_root / safe_brand / f"{stem}_foreground.png"
            target_mask = mask_root / safe_brand / f"{stem}_target.png"
            width, height = int(rec["width"]), int(rec["height"])
            mask_image([], width, height, query_mask, foreground=True)
            mask_image(rec.get("boxes", []), width, height, target_mask, foreground=False)
            rows.append({
                "brand": safe_brand,
                "can_id": str(rec["can_id"]),
                "query": str(image_path.resolve()),
                "query_mask": str(query_mask.resolve()),
                "target_mask": str(target_mask.resolve()),
                "label": int(rec["label_id"]),
            })

    by_brand = {}
    for row in rows:
        by_brand.setdefault(row["brand"], []).append(row)
    train, val = [], []
    for brand, brand_rows in sorted(by_brand.items()):
        groups = sorted({row["can_id"] for row in brand_rows})
        train_group_count = max(1, int(len(groups) * 0.7))
        train_groups = set(groups[:train_group_count])
        train_rows = [row for row in brand_rows if row["can_id"] in train_groups]
        val_rows = [row for row in brand_rows if row["can_id"] not in train_groups]
        refs = [row for row in train_rows if row["label"] == 0]
        if not refs:
            refs = [row for row in brand_rows if row["label"] == 0]
        refs = refs[:4]
        if not refs:
            raise RuntimeError(f"no normal reference image for {brand}")
        for row in train_rows + val_rows:
            row = dict(row)
            row_refs = [ref for ref in refs if ref["query"] != row["query"]]
            if not row_refs:
                row_refs = list(refs)
            while len(row_refs) < 4:
                row_refs.append(refs[len(row_refs) % len(refs)])
            row_refs = row_refs[:4]
            row["references"] = [ref["query"] for ref in row_refs]
            row["reference_masks"] = [ref["query_mask"] for ref in row_refs]
            (train if row["can_id"] in train_groups else val).append(row)

    output.mkdir(parents=True, exist_ok=True)
    for name, records in (("train", train), ("val", val), ("all", rows)):
        (output / f"{name}.jsonl").write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
            encoding="utf-8",
        )
    summary = {
        "total": len(rows),
        "train": len(train),
        "val": len(val),
        "brands": {brand: sum(row["brand"] == brand for row in rows) for brand in sorted(by_brand)},
        "anomalies": sum(row["label"] == 1 for row in rows),
        "normals": sum(row["label"] == 0 for row in rows),
        "foreground_mask": "fixed central rectangle x=[0.20,0.80], y=[0.10,0.95]",
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
