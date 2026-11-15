"""本地 DINOv2 加载和图像掩码预处理。"""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch
from torch import nn


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


def load_rgb(path, image_size):
    image = Image.open(path).convert("RGB")
    image = image.resize((image_size, image_size), getattr(Image, "Resampling", Image).BICUBIC)
    image = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255
    # 使用 DINOv2 的 ImageNet 归一化参数。
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    return (image - mean) / std


def load_mask(path, image_size):
    # 掩码使用最近邻缩放，避免产生灰色边界。
    mask = Image.open(path).convert("L")
    mask = mask.resize((image_size, image_size), getattr(Image, "Resampling", Image).NEAREST)
    return torch.from_numpy(np.asarray(mask).copy()).float().div(255).unsqueeze(0)
