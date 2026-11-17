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
    dice = dice_loss.mean()
    # 整图 BCE 权重 0.5。记录的 image 是加权后的值，三项相加等于 total。
    image = 0.5 * F.binary_cross_entropy_with_logits(image_logits, labels)
    total = pixel_loss + dice + image
    return total, pixel_loss, dice, image


def image_auroc(labels, scores):
    # 正样本分数高于负样本的比例，并列记一半。验证集只有两百多张。
    positive = scores[labels > 0.5]
    negative = scores[labels <= 0.5]
    above = (positive[:, None] > negative[None, :]).sum()
    tied = (positive[:, None] == negative[None, :]).sum()
    return ((above + 0.5 * tied) / (positive.numel() * negative.numel())).item()


def validation_batch(pixel_logits, image_logits, target_mask, query_mask, labels):
    # 图像分数与模型一致：前景最高 1% 像素 logit 的平均，再做 sigmoid。
    score = image_logits.detach().sigmoid()
    probability = pixel_logits.detach().sigmoid()
    positive = labels.detach() > 0.5
    defect = target_mask.detach() > 0.5
    foreground = query_mask.detach() >= 0.5
    positive_defect = defect & positive[:, None, None, None]
    other_foreground = foreground & ~defect
    height, width = pixel_logits.shape[-2:]
    top_count = height * width // 100
    ranked = pixel_logits.detach().masked_fill(query_mask < 0.5, -20)
    chosen = ranked.flatten(1).topk(top_count, dim=1).indices
    hit = defect.flatten(1).gather(1, chosen).float().mean(dim=1)
    return {
        "score": score.float().cpu(),
        "label": labels.detach().float().cpu(),
        "defect_sum": positive_defect.float().mul(probability).sum().item(),
        "defect_count": int(positive_defect.sum().item()),
        "foreground_sum": other_foreground.float().mul(probability).sum().item(),
        "foreground_count": int(other_foreground.sum().item()),
        "topk_hit": hit[positive].sum().item(),
        "topk_count": int(positive.sum().item()),
    }


def summarize_validation(batches):
    scores = torch.cat([batch["score"] for batch in batches])
    labels = torch.cat([batch["label"] for batch in batches])
    positive = labels > 0.5
    defect_count = sum(batch["defect_count"] for batch in batches)
    foreground_count = sum(batch["foreground_count"] for batch in batches)
    topk_count = sum(batch["topk_count"] for batch in batches)
    return {
        "val_pos_score": scores[positive].mean().item(),
        "val_neg_score": scores[~positive].mean().item(),
        "val_image_auroc": image_auroc(labels, scores),
        "val_defect_prob": sum(batch["defect_sum"] for batch in batches) / defect_count,
        "val_foreground_prob": (
            sum(batch["foreground_sum"] for batch in batches) / foreground_count
        ),
        "val_topk_hit": sum(batch["topk_hit"] for batch in batches) / topk_count,
    }


def run_epoch(model, loader, optimizer, device, training):
    # training 为 False 时模型处于 eval，并且下面的反传会被跳过。
    model.train(training)
    total_loss = 0.0
    pixel_sum = 0.0
    dice_sum = 0.0
    image_sum = 0.0
    observed = []
    for batch in loader:
        # batch 顺序与 ReferenceDataset 的返回顺序一致。
        query, references, query_mask, reference_masks, target_mask, labels = [
            item.to(device) for item in batch
        ]
        with torch.set_grad_enabled(training):
            pixel_logits, image_logits = model(
                query, references, query_mask, reference_masks
            )
            loss, pixel, dice, image = loss_fn(
                pixel_logits, image_logits, target_mask, query_mask, labels
            )
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                observed.append(
                    validation_batch(
                        pixel_logits, image_logits, target_mask, query_mask, labels
                    )
                )
        # 按 batch 内样本数累加，最后再除以整个数据集的样本数。
        count = query.shape[0]
        total_loss += loss.item() * count
        pixel_sum += pixel.item() * count
        dice_sum += dice.item() * count
        image_sum += image.item() * count
    count = len(loader.dataset)
    losses = (
        total_loss / count,
        pixel_sum / count,
        dice_sum / count,
        image_sum / count,
    )
    if training:
        return losses
    return losses, summarize_validation(observed)


def fit(model, train_loader, val_loader, optimizer, device, epochs, save_dir, log_dir):
    save_dir = Path(save_dir)
    log_dir = Path(log_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    history = []
    best_loss = float("inf")
    for epoch in range(epochs):
        # 每个 epoch 先训练，再用验证集选择最优权重。
        train_loss, train_pixel, train_dice, train_image = run_epoch(
            model, train_loader, optimizer, device, True
        )
        (val_loss, val_pixel, val_dice, val_image), observed = run_epoch(
            model, val_loader, optimizer, device, False
        )
        record = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_pixel": train_pixel,
            "train_dice": train_dice,
            "train_image": train_image,
            "val_loss": val_loss,
            "val_pixel": val_pixel,
            "val_dice": val_dice,
            "val_image": val_image,
        }
        record.update(observed)
        history.append(record)
        print(
            f"epoch={epoch + 1}/{epochs} "
            f"train={train_loss:.6f} pixel={train_pixel:.6f} "
            f"dice={train_dice:.6f} image={train_image:.6f} "
            f"val={val_loss:.6f} pixel={val_pixel:.6f} "
            f"dice={val_dice:.6f} image={val_image:.6f}",
            flush=True,
        )
        print(
            f"val_pos_score={observed['val_pos_score']:.6f} "
            f"val_neg_score={observed['val_neg_score']:.6f} "
            f"val_image_auroc={observed['val_image_auroc']:.6f} "
            f"val_defect_prob={observed['val_defect_prob']:.6f} "
            f"val_foreground_prob={observed['val_foreground_prob']:.6f} "
            f"val_topk_hit={observed['val_topk_hit']:.6f}",
            flush=True,
        )
        # 验证总损失下降时覆盖 best.pt，训练结束后用这份权重做预测。
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
