"""从 train.yaml 读取训练参数。"""

from dataclasses import dataclass
from pathlib import Path

import yaml


# main.py 所在目录。yaml 里的相对路径相对这里解析。
ROOT = Path(__file__).resolve().parents[1]


def _resolve(path):
    # 空权重表示使用仓库里的 safetensors。绝对路径保持不变。
    if path is None:
        return None
    path = Path(path)
    if path.is_absolute():
        return path
    return ROOT / path


@dataclass
class TrainConfig:
    train_manifest: Path
    val_manifest: Path
    dino_repo: Path
    dino_weights: Path | None
    image_size: int
    batch_size: int
    epochs: int
    learning_rate: float
    device: str
    seed: int
    save_dir: Path
    log_dir: Path
    output_dir: Path

    @classmethod
    def load(cls, path):
        with Path(path).open(encoding="utf-8") as config_file:
            raw = yaml.safe_load(config_file)
        return cls(
            train_manifest=_resolve(raw["train_manifest"]),
            val_manifest=_resolve(raw["val_manifest"]),
            dino_repo=_resolve(raw["dino_repo"]),
            dino_weights=_resolve(raw["dino_weights"]),
            image_size=raw["image_size"],
            batch_size=raw["batch_size"],
            epochs=raw["epochs"],
            learning_rate=raw["learning_rate"],
            device=raw["device"],
            seed=raw["seed"],
            save_dir=_resolve(raw["save_dir"]),
            log_dir=_resolve(raw["log_dir"]),
            output_dir=_resolve(raw["output_dir"]),
        )
