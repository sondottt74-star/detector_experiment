"""DINOv2 加载、图像掩码读取和异常热力图导出。"""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch import nn


# load_rgb 和热力图反归一化共用这一组 ImageNet 参数。
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# 低异常概率偏蓝，高异常概率偏红。
_HEATMAP_COLORS = np.array(
    [
        [0.0, 0.0, 0.5],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=np.float32,
)
_HEATMAP_STOPS = np.linspace(0.0, 1.0, len(_HEATMAP_COLORS))


class LocalDINOv2(nn.Module):
    def __init__(self, network):
        super().__init__()
        self.network = network

    def forward(self, pixel_values):
        # 将官方 DINOv2 的输出整理成 Detector 使用的 last_hidden_state。
        features = self.network.forward_features(pixel_values)
        tokens = torch.cat(
            (features["x_norm_clstoken"][:, None], features["x_norm_patchtokens"]),
            dim=1,
        )
        return SimpleNamespace(last_hidden_state=tokens)


def load_local_dinov2(repo_dir, weights_path=None):
    # 优先加载 ModelScope 的 HuggingFace 格式快照，始终保持离线。
    repo_dir = Path(repo_dir)
    if (repo_dir / "config.json").exists() and (repo_dir / "model.safetensors").exists():
        from transformers import AutoModel

        return AutoModel.from_pretrained(str(repo_dir), local_files_only=True)

    # 兼容原仓库约定的官方 DINOv2 代码和 .pth 权重。
    network = torch.hub.load(
        str(repo_dir),
        "dinov2_vits14",
        source="local",
        pretrained=False,
    )
    weights = torch.load(weights_path, map_location="cpu", weights_only=True)
    network.load_state_dict(weights)
    return LocalDINOv2(network)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


_RESAMPLE = getattr(Image, "Resampling", Image)
_MEAN_U8 = tuple(int(round(channel * 255)) for channel in IMAGENET_MEAN)


def fit_foreground(image, mask, size=518, margin=14):
    """按前景外接框等比例放入 size×size，框外和补齐处用 ImageNet 均值。"""
    image_array = np.asarray(image.convert("RGB"))
    mask_array = np.asarray(mask.convert("L")) > 127
    ys, xs = np.where(mask_array)
    # 外接框每边再留 margin，避免罐沿被切掉。坐标可以超出原图。
    x0 = int(xs.min()) - margin
    y0 = int(ys.min()) - margin
    x1 = int(xs.max()) + margin
    y1 = int(ys.max()) + margin
    crop_w = x1 - x0 + 1
    crop_h = y1 - y0 + 1
    height, width = mask_array.shape
    mean = np.array(_MEAN_U8, dtype=np.uint8)
    # 先铺满均值，再把落在原图内的那一块贴进来。超出原图的边自然留成均值。
    crop = np.empty((crop_h, crop_w, 3), dtype=np.uint8)
    crop[:] = mean
    crop_mask = np.zeros((crop_h, crop_w), dtype=bool)
    src_x0, src_y0 = max(x0, 0), max(y0, 0)
    src_x1, src_y1 = min(x1 + 1, width), min(y1 + 1, height)
    dst_x0, dst_y0 = src_x0 - x0, src_y0 - y0
    crop[dst_y0:dst_y0 + src_y1 - src_y0, dst_x0:dst_x0 + src_x1 - src_x0] = (
        image_array[src_y0:src_y1, src_x0:src_x1]
    )
    crop_mask[dst_y0:dst_y0 + src_y1 - src_y0, dst_x0:dst_x0 + src_x1 - src_x0] = (
        mask_array[src_y0:src_y1, src_x0:src_x1]
    )
    # 框内不属于前景的像素也换成均值，背景不进 DINOv2。
    crop[~crop_mask] = mean

    # 宽高共用一个倍率，长边正好等于 size。
    scale = size / max(crop_w, crop_h)
    resized_w = int(round(crop_w * scale))
    resized_h = int(round(crop_h * scale))
    if crop_h >= crop_w:
        resized_h = size
    else:
        resized_w = size
    resized = np.array(
        Image.fromarray(crop).resize((resized_w, resized_h), _RESAMPLE.BICUBIC)
    )
    # 掩码用最近邻，避免缩放后出现灰色边缘。
    resized_mask = np.array(
        Image.fromarray(crop_mask.astype(np.uint8) * 255).resize(
            (resized_w, resized_h), _RESAMPLE.NEAREST
        )
    ) > 127
    resized[~resized_mask] = mean

    # 短边居中补到正方形。补齐区域掩码保持 0，不参与对齐和计分。
    canvas = np.full((size, size, 3), mean, dtype=np.uint8)
    canvas_mask = np.zeros((size, size), dtype=np.uint8)
    left = (size - resized_w) // 2
    top = (size - resized_h) // 2
    canvas[top:top + resized_h, left:left + resized_w] = resized
    canvas_mask[top:top + resized_h, left:left + resized_w][resized_mask] = 255
    # window 供缺陷框使用：原图坐标先减去 (x0, y0)，再乘 scale，最后加上补齐偏移。
    return Image.fromarray(canvas), Image.fromarray(canvas_mask), (x0, y0, scale, left, top)


def load_rgb(path):
    # 图在划分数据集时已经是正方形，这里只做 ImageNet 归一化。
    image = Image.open(path).convert("RGB")
    image = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255
    mean = torch.tensor(IMAGENET_MEAN)[:, None, None]
    std = torch.tensor(IMAGENET_STD)[:, None, None]
    return (image - mean) / std


def load_mask(path):
    # 掩码与图一起写好，读取时不再缩放。
    mask = Image.open(path).convert("L")
    return torch.from_numpy(np.asarray(mask).copy()).float().div(255).unsqueeze(0)


def colorize(score):
    """把 0 到 1 的异常概率映射成 RGB。"""
    flat = np.clip(score, 0.0, 1.0).reshape(-1)
    channels = [
        np.interp(flat, _HEATMAP_STOPS, _HEATMAP_COLORS[:, channel])
        for channel in range(3)
    ]
    return np.stack(channels, axis=1).reshape(*score.shape, 3)


def denormalize(image):
    """把送进 DINOv2 的张量还原成可显示的 RGB 图。"""
    array = image.detach().float().cpu().permute(1, 2, 0).numpy()
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)
    return np.clip(array * std + mean, 0.0, 1.0)


def _spatial(value):
    """去掉通道维，得到一张 H×W 图。张量和数组都能进来。"""
    if torch.is_tensor(value):
        value = value.detach().float().cpu().numpy()
    else:
        value = np.asarray(value)
    if value.ndim == 3:
        value = value[0]
    return value


def _panel(image):
    return (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)


def render_heatmap(image, score, anomaly_score, target=None):
    """从左到右拼出原图、预测叠加、热力图和缺陷标注。"""
    score = _spatial(score)
    original = denormalize(image)
    color = colorize(score)
    # 概率同时作为透明度：背景和低分区域保留原图，高分区域显示红色。
    weight = np.clip(score, 0.0, 1.0)[..., None]
    overlay = original * (1.0 - 0.65 * weight) + color * (0.65 * weight)
    panels = [original, overlay, color]
    if target is not None:
        # 缺陷标注用绿色盖在原图上，和红色预测分开。正常图没有缺陷，这一张保持原图。
        truth = _spatial(target) > 0.5
        green = np.zeros_like(original)
        green[..., 1] = 1.0
        cover = truth.astype(np.float32)[..., None]
        panels.append(original * (1.0 - 0.65 * cover) + green * (0.65 * cover))
    canvas = Image.fromarray(np.concatenate([_panel(panel) for panel in panels], axis=1))
    draw = ImageDraw.Draw(canvas)
    text = f"score {float(anomaly_score):.3f}"
    text_box = draw.textbbox((4, 2), text)
    draw.rectangle((0, 0, text_box[2] + 4, text_box[3] + 2), fill=(0, 0, 0))
    draw.text((4, 2), text, fill=(255, 255, 255))
    if target is not None:
        left = original.shape[1] * 3 + 4
        gt_box = draw.textbbox((left, 2), "gt")
        draw.rectangle((left - 4, 0, gt_box[2] + 4, gt_box[3] + 2), fill=(0, 0, 0))
        draw.text((left, 2), "gt", fill=(255, 255, 255))
    return canvas


def save_heatmap_image(image, score, anomaly_score, path, target=None):
    """把一张待测图的热力图写成 PNG。target 是同一分辨率的缺陷掩码。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    render_heatmap(image, score, anomaly_score, target).save(path)
