"""Detector 实验入口。"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from detector.dataset import ReferenceDataset
from detector.model import Detector
from detector.trainer import fit, save_predictions
from detector.utils import load_local_dinov2


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--dino-repo", required=True)
    parser.add_argument("--dino-weights", default=None)
    parser.add_argument("--image-size", type=int, default=518)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.manual_seed(0)

    train_set = ReferenceDataset(args.train_manifest, args.image_size)
    val_set = ReferenceDataset(args.val_manifest, args.image_size)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    # DINOv2 从本地官方代码和权重加载，Detector 只训练新增模块。
    encoder = load_local_dinov2(args.dino_repo, args.dino_weights)
    model = Detector(encoder).to(args.device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
    )

    fit(
        model,
        train_loader,
        val_loader,
        optimizer,
        args.device,
        args.epochs,
        ROOT / "save",
        ROOT / "log",
    )
    model.load_state_dict(torch.load(ROOT / "save/best.pt", map_location=args.device))
    save_predictions(model, val_loader, args.device, ROOT / "output")


if __name__ == "__main__":
    main()
