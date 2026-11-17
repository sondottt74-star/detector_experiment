"""根据测试清单的 predictions.jsonl 和热力图计算图像级、像素级指标。"""

import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.train_config import resolve
from detector.utils import load_mask


def load_sklearn():
    # 项目虚拟环境没有装 sklearn，使用用户已经安装的那一份。
    try:
        import sklearn.metrics as metrics
    except ModuleNotFoundError:
        user_site = Path.home() / ".local/lib/python3.10/site-packages"
        if str(user_site) not in sys.path:
            sys.path.append(str(user_site))
        import sklearn.metrics as metrics
    return metrics


def safe_score(function, labels, scores):
    if len(np.unique(labels)) < 2:
        return None
    return float(function(labels, scores))


def classification_at_half(metrics, labels, scores):
    predicted = (scores >= 0.5).astype(np.int32)
    return {
        "precision_at_0.5": float(metrics.precision_score(labels, predicted, zero_division=0)),
        "recall_at_0.5": float(metrics.recall_score(labels, predicted, zero_division=0)),
        "f1_at_0.5": float(metrics.f1_score(labels, predicted, zero_division=0)),
    }


def image_metrics(metrics, rows):
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int32)
    scores = np.asarray([float(row["score"]) for row in rows], dtype=np.float64)
    result = {
        "n": int(labels.size),
        "n_positive": int(labels.sum()),
        "image_auroc": safe_score(metrics.roc_auc_score, labels, scores),
        "image_ap": safe_score(metrics.average_precision_score, labels, scores),
    }
    result.update(classification_at_half(metrics, labels, scores))
    per_tag = {}
    tags = sorted({row["tag"] for row in rows})
    for tag in tags:
        chosen = [row for row in rows if row["tag"] == tag]
        tag_labels = np.asarray([int(row["label"]) for row in chosen], dtype=np.int32)
        tag_scores = np.asarray([float(row["score"]) for row in chosen], dtype=np.float64)
        per_tag[tag] = {
            "n": int(tag_labels.size),
            "n_positive": int(tag_labels.sum()),
            "image_auroc": safe_score(metrics.roc_auc_score, tag_labels, tag_scores),
            "image_ap": safe_score(metrics.average_precision_score, tag_labels, tag_scores),
        }
    result["per_tag"] = per_tag
    return result


def pixel_metrics(metrics, records, output_dir, batch_size):
    scores = []
    labels = []
    offset = 0
    batch_count = (len(records) + batch_size - 1) // batch_size
    for batch_index in range(batch_count):
        path = output_dir / f"{batch_index:04d}_heatmap.npy"
        if not path.is_file():
            return {"pixel_auroc": None, "pixel_ap": None, "n_pixels": 0}
        heatmap = np.load(path)
        spatial = heatmap[:, 0] if heatmap.ndim == 4 else heatmap
        for local in range(len(spatial)):
            record = records[offset + local]
            query_mask = load_mask(record["query_mask"]).numpy()[0] > 0.5
            target = load_mask(record["target_mask"]).numpy()[0] > 0.5
            if query_mask.shape != spatial[local].shape:
                raise SystemExit(f"heatmap shape mismatch: {path}")
            if query_mask.any():
                scores.append(spatial[local][query_mask].astype(np.float32, copy=False))
                labels.append(target[query_mask])
        offset += len(spatial)
    if offset != len(records) or not scores:
        return {"pixel_auroc": None, "pixel_ap": None, "n_pixels": 0}
    flat_scores = np.concatenate(scores)
    flat_labels = np.concatenate(labels).astype(np.int32)
    return {
        "n_pixels": int(flat_labels.size),
        "n_positive_pixels": int(flat_labels.sum()),
        "pixel_auroc": safe_score(metrics.roc_auc_score, flat_labels, flat_scores),
        "pixel_ap": safe_score(metrics.average_precision_score, flat_labels, flat_scores),
    }


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: evaluate.py CONFIG.yaml")
    config_path = Path(sys.argv[1])
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    manifest = resolve(raw["infer_manifest"])
    output_dir = resolve(raw["infer_output"])
    prediction_path = output_dir / "predictions.jsonl"
    if not prediction_path.is_file():
        raise SystemExit(f"missing predictions: {prediction_path}")
    rows = [json.loads(line) for line in prediction_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    metrics = load_sklearn()
    result = image_metrics(metrics, rows)
    result.update(pixel_metrics(metrics, records, output_dir, int(raw["batch_size"])))
    result["manifest"] = str(manifest)
    result["predictions"] = str(prediction_path)
    destination = output_dir / "metrics.json"
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"test n={result['n']} image_auroc={result['image_auroc']} "
        f"image_ap={result['image_ap']} pixel_auroc={result['pixel_auroc']} "
        f"-> {destination}",
        flush=True,
    )


if __name__ == "__main__":
    main()
