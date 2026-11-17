"""按一份 yaml 训练，并在测试清单上推理、计分。

训练步骤与 main.py 相同。这里在训练后继续推理和计分，
方便五折和 zeroshot 各写各的目录。
"""

import json
import subprocess
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.train_config import TrainConfig, resolve
from detector.dataset import ReferenceDataset
from detector.model import Detector
from detector.trainer import fit
from detector.utils import load_local_dinov2, read_jsonl


def manifest_count(path):
    rows = read_jsonl(path)
    if not rows:
        raise SystemExit(f"empty manifest: {path}")
    sample = rows[0]
    required = [sample["query"], sample["query_mask"], sample["target_mask"]]
    required.extend(sample["references"])
    required.extend(sample["reference_masks"])
    missing = [item for item in required if not Path(item).is_file()]
    if missing:
        raise SystemExit(f"missing file: {missing[0]}")
    return len(rows)


def training_done(config):
    history_path = config.log_dir / "history.json"
    checkpoint = config.save_dir / "best.pt"
    if not history_path.is_file() or not checkpoint.is_file():
        return False
    try:
        history = json.loads(history_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return len(history) >= config.epochs


def predictions_done(manifest, output_dir):
    path = Path(output_dir) / "predictions.jsonl"
    if not path.is_file():
        return False
    predicted = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return len(predicted) == manifest_count(manifest)


def train(config):
    torch.manual_seed(config.seed)
    train_set = ReferenceDataset(
        config.train_manifest, config.image_size, config.augment, config.reference_count
    )
    val_set = ReferenceDataset(
        config.val_manifest, config.image_size, reference_count=config.reference_count
    )
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=False)

    encoder = load_local_dinov2(config.dino_repo, config.dino_weights)
    model = Detector(encoder).to(config.device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.learning_rate,
    )
    print(
        f"train={len(train_set)} val={len(val_set)} "
        f"batch={config.batch_size} epochs={config.epochs} device={config.device}",
        flush=True,
    )
    if config.device.startswith("cuda"):
        print(f"gpu={torch.cuda.get_device_name(config.device)}", flush=True)

    fit(
        model,
        train_loader,
        val_loader,
        optimizer,
        config.device,
        config.epochs,
        config.save_dir,
        config.log_dir,
    )


def extra_paths(config_path, config):
    raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    manifest = resolve(raw.get("infer_manifest")) or config.val_manifest
    output_dir = resolve(raw.get("infer_output")) or config.output_dir
    return manifest, output_dir


def run_child(script, config_path):
    subprocess.run(
        [sys.executable, str(ROOT / script), str(config_path)],
        cwd=ROOT,
        check=True,
    )


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: run_one.py CONFIG.yaml")
    config_path = Path(sys.argv[1]).resolve()
    config = TrainConfig.load(config_path)
    manifest_count(config.train_manifest)
    manifest_count(config.val_manifest)
    test_manifest, test_output = extra_paths(config_path, config)
    manifest_count(test_manifest)

    print(f"config={config_path}", flush=True)
    if training_done(config):
        print(f"skip train, found {config.save_dir / 'best.pt'}", flush=True)
    else:
        train(config)

    if predictions_done(test_manifest, test_output):
        print(f"skip test, found {test_output / 'predictions.jsonl'}", flush=True)
    else:
        run_child("inference.py", config_path)
    run_child("script/evaluate.py", config_path)


if __name__ == "__main__":
    main()
