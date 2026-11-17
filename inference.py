"""用训练好的 Detector 对一份清单做推理，不更新权重。"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from config.train_config import TrainConfig
from detector.dataset import ReferenceDataset
from detector.model import Detector
from detector.utils import load_local_dinov2, read_jsonl, save_heatmap_image


ROOT = Path(__file__).resolve().parent


def resolve(path):
    # 与 TrainConfig 相同：相对路径相对 detector_experiment。
    if path is None:
        return None
    path = Path(path)
    if path.is_absolute():
        return path
    return ROOT / path


def predict(model, loader, records, device, output_dir):
    """逐张写出热力图，并按清单顺序保存分数。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    predictions = []
    sample_index = 0
    with torch.no_grad():
        for index, batch in enumerate(loader):
            query, references, query_mask, reference_masks, target_mask, _ = [
                item.to(device) for item in batch
            ]
            pixel_logits, image_logits = model(
                query, references, query_mask, reference_masks
            )
            # 背景不显示异常。整图分数是前景最高 1% 像素的平均。
            heatmap = pixel_logits.sigmoid() * query_mask
            score = image_logits.sigmoid()
            np.save(output_dir / f"{index:04d}_heatmap.npy", heatmap.cpu().numpy())
            np.save(output_dir / f"{index:04d}_score.npy", score.cpu().numpy())
            for query_image, sample_heatmap, sample_score, sample_target in zip(
                query, heatmap, score, target_mask
            ):
                record = records[sample_index]
                value = float(sample_score)
                save_heatmap_image(
                    query_image,
                    sample_heatmap,
                    value,
                    output_dir / f"{sample_index:04d}_heatmap.png",
                    sample_target,
                )
                predictions.append({
                    "index": sample_index,
                    "score": value,
                    "label": record["label"],
                    "tag": record["tag"],
                    "query": record["query"],
                })
                sample_index += 1
    (output_dir / "predictions.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions),
        encoding="utf-8",
    )
    print(f"wrote {output_dir / 'predictions.jsonl'} n={len(predictions)}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    config_path = resolve(args.config)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config = TrainConfig.load(config_path)
    # 不写 infer_manifest 时，对这份配置的验证清单推理。
    manifest = resolve(raw.get("infer_manifest")) or config.val_manifest
    checkpoint = resolve(raw.get("checkpoint")) or (config.save_dir / "best.pt")
    output_dir = resolve(raw.get("infer_output")) or config.output_dir

    records = read_jsonl(manifest)
    dataset = ReferenceDataset(manifest, config.image_size)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)

    encoder = load_local_dinov2(config.dino_repo, config.dino_weights)
    model = Detector(encoder).to(config.device)
    model.load_state_dict(torch.load(checkpoint, map_location=config.device))
    predict(model, loader, records, config.device, output_dir)


if __name__ == "__main__":
    main()
