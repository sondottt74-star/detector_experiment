"""Detector 的训练、验证和预测输出。"""

import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


def loss_fn(pixel_logits, image_logits, target_mask, query_mask, labels):
    # 像素 BCE、区域 Dice 和整图分类共同训练检测器。
    pixel_loss = F.binary_cross_entropy_with_logits(
        pixel_logits, target_mask, reduction="none"
    )
    pixel_loss = (pixel_loss * query_mask).sum() / query_mask.sum()
    probability = pixel_logits.sigmoid() * query_mask
    truth = target_mask * query_mask
    dice_loss = 1 - (
        2 * (probability * truth).sum((1, 2, 3)) + 1
    ) / (probability.sum((1, 2, 3)) + truth.sum((1, 2, 3)) + 1)
    image_loss = F.binary_cross_entropy_with_logits(image_logits, labels)
    return pixel_loss + dice_loss.mean() + 0.5 * image_loss


def run_epoch(model, loader, optimizer, device, training):
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
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), save_dir / "best.pt")

    (log_dir / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2)
    )
    return history


def save_predictions(model, loader, device, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for index, batch in enumerate(loader):
            # 保存像素热力图和整图异常分数，供后续可视化或评估使用。
            query, references, query_mask, reference_masks, _, _ = [
                item.to(device) for item in batch
            ]
            pixel_logits, image_logits = model(
                query, references, query_mask, reference_masks
            )
            heatmap = pixel_logits.sigmoid() * query_mask
            np.save(output_dir / f"{index:04d}_heatmap.npy", heatmap.cpu().numpy())
            np.save(output_dir / f"{index:04d}_score.npy", image_logits.sigmoid().cpu().numpy())
