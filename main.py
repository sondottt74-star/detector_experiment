"""Detector 实验入口。"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from config.train_config import TrainConfig
from detector.dataset import ReferenceDataset
from detector.model import Detector
from detector.trainer import fit, save_predictions
from detector.utils import load_local_dinov2


ROOT = Path(__file__).resolve().parent


def main():
    config = TrainConfig.load(ROOT / "config" / "train.yaml")
    torch.manual_seed(config.seed)

    train_set = ReferenceDataset(config.train_manifest, config.image_size)
    val_set = ReferenceDataset(config.val_manifest, config.image_size)
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=False)

    # DINOv2 从本地官方代码和权重加载，Detector 只训练新增模块。
    encoder = load_local_dinov2(config.dino_repo, config.dino_weights)
    model = Detector(encoder).to(config.device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.learning_rate,
    )

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
    model.load_state_dict(torch.load(config.save_dir / "best.pt", map_location=config.device))
    save_predictions(model, val_loader, config.device, config.output_dir)


if __name__ == "__main__":
    main()
