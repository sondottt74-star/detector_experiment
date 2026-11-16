"""从 SAM2 前景掩码和渲染参考库生成检测器清单。

每张待测图用 DINOv2 余弦相似度检索正常参考图，供 model.py 的交叉注意力使用。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from detector.utils import fit_foreground, load_local_dinov2, load_rgb


def write_fitted(image_path, mask_path, image_out, mask_out, size):
    """用前景掩码把一张图做成 size×size，并写出图和掩码。"""
    image, mask, window = fit_foreground(
        Image.open(image_path), Image.open(mask_path), size
    )
    image_out.parent.mkdir(parents=True, exist_ok=True)
    image.save(image_out, quality=95)
    mask.save(mask_out)
    return window


def write_target_mask(boxes, window, path, size):
    """把原图缺陷框映射到等比例正方形上，画成白色目标掩码。"""
    # window 来自 fit_foreground：裁剪原点、统一倍率、居中补齐的偏移。
    origin_x, origin_y, scale, left, top = window
    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)
    for box in boxes:
        x0, y0, x1, y1 = [float(value) for value in box]
        draw.rectangle(
            (
                (x0 - origin_x) * scale + left,
                (y0 - origin_y) * scale + top,
                (x1 - origin_x) * scale + left,
                (y1 - origin_y) * scale + top,
            ),
            fill=255,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    mask.save(path)


def encode_paths(encoder, paths, batch_size, device):
    """把一组图像编成检索向量：DINOv2 patch token 的均值，再做 L2 归一化。"""
    vectors = []
    encoder.eval()
    with torch.inference_mode():
        for start in range(0, len(paths), batch_size):
            images = torch.stack(
                [load_rgb(path) for path in paths[start:start + batch_size]]
            ).to(device)
            tokens = encoder(pixel_values=images).last_hidden_state
            # 丢掉 CLS（index 0），只平均空间 patch，避免整图语义盖过局部外观。
            vector = F.normalize(tokens[:, 1:].mean(dim=1), dim=-1)
            vectors.append(vector.cpu())
    return torch.cat(vectors, dim=0)


def load_bank(path):
    """读取 reference_bank.json 里按品牌分组的正常参考图。"""
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
    parser.add_argument("--image-size", type=int, default=518)
    parser.add_argument("--reference-count", type=int, default=4)
    parser.add_argument("--retrieval-batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    output = Path(args.output).resolve()
    prepared = output / "prepared"
    test_root = data_root / "00_raw" / "query_inputs"
    bank = load_bank(args.reference_bank)
    device = torch.device(args.device)
    encoder = load_local_dinov2(args.dino_repo, args.dino_weights).to(device)

    # 待测图用 SAM2 掩码做成正方形，缺陷框按同一个窗口换算。
    rows_by_brand = {}
    for brand_dir in sorted(p for p in test_root.iterdir() if p.is_dir()):
        payload = json.loads((brand_dir / "annotations.json").read_text(encoding="utf-8"))
        records = []
        for rec in sorted(payload["records"], key=lambda x: (x["can_id"], x["file"])):
            image_path = brand_dir / "images" / rec["file"]
            stem = Path(rec["file"]).stem
            query_mask = Path(args.sam2_mask_root) / brand_dir.name / f"{stem}_mask.png"
            query_dir = prepared / "queries" / brand_dir.name
            fitted_image = query_dir / f"{stem}.jpg"
            fitted_mask = query_dir / f"{stem}_mask.png"
            target_mask = query_dir / f"{stem}_target.png"
            window = write_fitted(image_path, query_mask, fitted_image, fitted_mask, args.image_size)
            write_target_mask(rec.get("boxes", []), window, target_mask, args.image_size)
            records.append({
                "brand": brand_dir.name,
                "can_id": str(rec["can_id"]),
                "query": str(fitted_image.resolve()),
                "query_mask": str(fitted_mask.resolve()),
                "target_mask": str(target_mask.resolve()),
                "label": int(rec["label_id"]),
            })
        rows_by_brand[brand_dir.name] = records

    retrieval_cache = output / "retrieval_cache"
    retrieval_cache.mkdir(parents=True, exist_ok=True)
    for brand, rows in sorted(rows_by_brand.items()):
        # tag 是这批待测图对应的参考库名字。检索只在这个库里进行。
        if brand not in bank:
            aliases = [name for name in bank if brand in name or name in brand]
            if len(aliases) != 1:
                raise KeyError(f"reference bank has no unambiguous entry for {brand}")
            tag = aliases[0]
        else:
            tag = brand
        bank_entry = bank[tag]
        for row in rows:
            row["tag"] = tag
        # 高斯渲染图用建库时的轮廓掩码做同一套正方形，不再跑 SAM2。
        gallery = []
        reference_dir = prepared / "references" / tag
        for item in bank_entry["references"]:
            source = Path(item["image"])
            fitted_image = reference_dir / source.name
            fitted_mask = reference_dir / f"{source.stem}_mask.png"
            write_fitted(source, item["foreground_mask"], fitted_image, fitted_mask, args.image_size)
            gallery.append({
                "image": str(fitted_image.resolve()),
                "foreground_mask": str(fitted_mask.resolve()),
            })
        gallery_paths = [item["image"] for item in gallery]
        # 参考库向量按品牌和输入尺寸缓存。待测图用已经对齐的 518 图编码。
        cache_path = retrieval_cache / f"{tag}_{args.image_size}.pt"
        if cache_path.exists():
            gallery_vectors = torch.load(cache_path, map_location="cpu", weights_only=True)
        else:
            gallery_vectors = encode_paths(
                encoder, gallery_paths, args.retrieval_batch_size, device
            )
            torch.save(gallery_vectors, cache_path)
        query_vectors = encode_paths(
            encoder, [row["query"] for row in rows], args.retrieval_batch_size, device
        )
        # 两边都已 L2 归一化，点积就是余弦相似度。
        similarity = query_vectors @ gallery_vectors.T
        for row, scores in zip(rows, similarity):
            # 只从 tag 对应的参考库里保留相似度最高的若干张正常参考图。
            count = min(args.reference_count, len(gallery))
            indices = scores.topk(count).indices.tolist()
            row["references"] = [gallery[index]["image"] for index in indices]
            row["reference_masks"] = [gallery[index]["foreground_mask"] for index in indices]
            row["reference_scores"] = [float(scores[index]) for index in indices]

    all_rows = [row for rows in rows_by_brand.values() for row in rows]
    train, val = [], []
    for brand, rows in sorted(rows_by_brand.items()):
        # 按罐划分，同一只罐的多张图不会拆到训练和验证两边。
        # 验证集优先放入异常图更多的罐，并至少留一只罐，约占该品牌 30%。
        groups = sorted({row["can_id"] for row in rows})
        anomaly_count = {group: sum(row["label"] == 1 for row in rows if row["can_id"] == group) for group in groups}
        val_groups = set(sorted(groups, key=lambda group: (-anomaly_count[group], group))[:max(1, int(len(groups) * .3))])
        train.extend(row for row in rows if row["can_id"] not in val_groups)
        val.extend(row for row in rows if row["can_id"] in val_groups)

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
