"""Detector 实验入口。配置路径由命令行传入。"""

import sys

import torch
from torch.utils.data import DataLoader

from config.train_config import TrainConfig, resolve
from detector.dataset import ReferenceDataset
from detector.model import Detector
from detector.trainer import fit
from detector.utils import load_local_dinov2


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: main.py CONFIG.yaml")
    config = TrainConfig.load(resolve(sys.argv[1]))
    torch.manual_seed(config.seed)

    train_set = ReferenceDataset(
        config.train_manifest, config.image_size, config.augment, config.reference_count
    )
    val_set = ReferenceDataset(
        config.val_manifest, config.image_size, reference_count=config.reference_count
    )
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


if __name__ == "__main__":
    main()
