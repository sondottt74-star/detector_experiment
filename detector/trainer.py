"""Detector 的训练、验证和预测输出。"""

import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .utils import save_heatmap_image


def loss_fn(pixel_logits, image_logits, target_mask, query_mask, labels):
    # 像素项用 focal 权重：预测对的像素权重接近 0，缺陷像素再乘 4。
    # 前景掩码把背景排除在像素损失之外。
    bce = F.binary_cross_entropy_with_logits(
        pixel_logits, target_mask, reduction="none"
    )
    probability = pixel_logits.sigmoid()
    correct = target_mask * probability + (1 - target_mask) * (1 - probability)
    pixel_loss = (
        ((1 - correct).pow(2) * bce * (1 + 3 * target_mask) * query_mask).sum()
        / query_mask.sum().clamp_min(1)
    )
    # Dice 用预测区域和缺陷区域的重叠约束形状，分子分母都加 1 避免除零。
    probability = probability * query_mask
    truth = target_mask * query_mask
    dice_loss = 1 - (
        2 * (probability * truth).sum((1, 2, 3)) + 1
    ) / (probability.sum((1, 2, 3)) + truth.sum((1, 2, 3)) + 1)
    # 整图 BCE 权重 0.5，让像素定位仍占主要梯度。
    image_loss = F.binary_cross_entropy_with_logits(image_logits, labels)
    return pixel_loss + dice_loss.mean() + 0.5 * image_loss


def run_epoch(model, loader, optimizer, device, training):
    # training 为 False 时模型处于 eval，并且下面的反传会被跳过。
    model.train(training)
    total_loss = 0.0
    for batch in loader:
        # batch 顺序与 ReferenceDataset 的返回顺序一致。
        query, references, query_mask, reference_masks, target_mask, labels = [
            item.to(device) for item in batch
        ]
        with torch.set_grad_enabled(training):
            pixel_logits, image_logits = model(
                query, references, query_mask, reference_masks
            )
            loss = loss_fn(
                pixel_logits, image_logits, target_mask, query_mask, labels
            )
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        # 按 batch 内样本数累加，最后再除以整个数据集的样本数。
        total_loss += loss.item() * query.shape[0]
    return total_loss / len(loader.dataset)


def fit(model, train_loader, val_loader, optimizer, device, epochs, save_dir, log_dir):
    save_dir = Path(save_dir)
    log_dir = Path(log_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    history = []
    best_loss = float("inf")
    for epoch in range(epochs):
        # 每个 epoch 先训练，再用验证集选择最优权重。
        train_loss = run_epoch(model, train_loader, optimizer, device, True)
        val_loss = run_epoch(model, val_loader, optimizer, device, False)
        record = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        history.append(record)
        print(
            f"epoch={epoch + 1}/{epochs} train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f}",
            flush=True,
        )
        # 验证损失下降时覆盖 best.pt，训练结束后用这份权重做预测。
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), save_dir / "best.pt")

    print(f"best_val_loss={best_loss:.6f}", flush=True)
    (log_dir / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2)
    )
    return history


def save_predictions(model, loader, device, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    sample_index = 0
    with torch.no_grad():
        for index, batch in enumerate(loader):
            # query 仍保留在 device 上，后面逐张还原成热力图。
            query, references, query_mask, reference_masks, target_mask, _ = [
                item.to(device) for item in batch
            ]
            pixel_logits, image_logits = model(
                query, references, query_mask, reference_masks
            )
            # 概率乘前景掩码，背景位置的热力图为 0。
            heatmap = pixel_logits.sigmoid() * query_mask
            score = image_logits.sigmoid()
            # npy 按 batch 保存，供数值评估；PNG 按单张图保存，供直接查看。
            np.save(output_dir / f"{index:04d}_heatmap.npy", heatmap.cpu().numpy())
            np.save(output_dir / f"{index:04d}_score.npy", score.cpu().numpy())
            for query_image, sample_heatmap, sample_score, sample_target in zip(
                query, heatmap, score, target_mask
            ):
                save_heatmap_image(
                    query_image,
                    sample_heatmap,
                    float(sample_score),
                    output_dir / f"{sample_index:04d}_heatmap.png",
                    sample_target,
                )
                sample_index += 1
