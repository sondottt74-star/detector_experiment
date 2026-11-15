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
        features = self.network.forward_features(pixel_values)
        tokens = torch.cat(
            (features["x_norm_clstoken"][:, None], features["x_norm_patchtokens"]),
            dim=1,
        )
        return SimpleNamespace(last_hidden_state=tokens)


def load_local_dinov2(repo_dir, weights_path):
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
    image = image.resize((image_size, image_size), Image.Resampling.BICUBIC)
    image = torch.from_numpy(np.asarray(image)).permute(2, 0, 1).float() / 255
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    return (image - mean) / std


def load_mask(path, image_size):
    mask = Image.open(path).convert("L")
    mask = mask.resize((image_size, image_size), Image.Resampling.NEAREST)
    return torch.from_numpy(np.asarray(mask)).float().div(255).unsqueeze(0)
