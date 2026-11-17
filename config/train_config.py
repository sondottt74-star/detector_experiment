"""训练配置。相对路径相对 detector_experiment 目录。"""

from dataclasses import dataclass
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def resolve(path):
    if path is None:
        return None
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


@dataclass
class TrainConfig:
    seed: int
    train_manifest: Path
    val_manifest: Path
    image_size: int
    batch_size: int
    dino_repo: Path
    dino_weights: Path | None
    device: str
    learning_rate: float
    epochs: int
    save_dir: Path
    log_dir: Path
    output_dir: Path
    augment: bool
    reference_count: int

    @classmethod
    def load(cls, path):
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls(
            seed=int(raw["seed"]),
            train_manifest=resolve(raw["train_manifest"]),
            val_manifest=resolve(raw["val_manifest"]),
            image_size=int(raw["image_size"]),
            batch_size=int(raw["batch_size"]),
            dino_repo=resolve(raw["dino_repo"]),
            dino_weights=resolve(raw.get("dino_weights")),
            device=str(raw["device"]),
            learning_rate=float(raw["learning_rate"]),
            epochs=int(raw["epochs"]),
            save_dir=resolve(raw["save_dir"]),
            log_dir=resolve(raw["log_dir"]),
            output_dir=resolve(raw["output_dir"]),
            augment=bool(raw.get("augment", False)),
            reference_count=int(raw.get("reference_count", 4)),
        )
