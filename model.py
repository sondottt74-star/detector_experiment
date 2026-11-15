"""参考图条件异常检测器：DINOv2、交叉注意力和像素级检测头。"""

import torch
from torch import nn
from torch.nn import functional as F


class Detector(nn.Module):
    """Frozen DINOv2, reference cross attention and anomaly prediction head."""

    def __init__(self, encoder, feature_dim=384, hidden_dim=128):
        super().__init__()
        # DINOv2 只提取冻结的视觉特征，训练投影层、注意力层和检测头。
        self.encoder = encoder.requires_grad_(False).eval()
        self.project = nn.Linear(feature_dim, hidden_dim)
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, num_heads=4, batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Conv2d(hidden_dim * 3, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 1, 1),
        )

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    def encode(self, images):
        with torch.no_grad():
            tokens = self.encoder(pixel_values=images).last_hidden_state
        # 第一个 token 是 CLS，只保留空间 patch token。
        return self.project(tokens[:, 1:])

    def forward(self, query, references, query_mask, reference_masks):
        batch_size, reference_count, channels, height, width = references.shape
        grid_height = height // 14
        grid_width = width // 14
        token_count = grid_height * grid_width

        # 待测图和参考图分别编码，参考图的 patch token 合并到同一序列。
        query_tokens = self.encode(query)
        reference_tokens = self.encode(
            references.reshape(batch_size * reference_count, channels, height, width)
        )
        reference_tokens = reference_tokens.reshape(
            batch_size, reference_count * token_count, -1
        )

        reference_valid = F.adaptive_max_pool2d(
            reference_masks.reshape(batch_size * reference_count, 1, height, width),
            (grid_height, grid_width),
        )
        reference_valid = reference_valid.reshape(
            batch_size, reference_count * token_count
        ) > 0

        # 用待测图作为 Query，用正常参考图作为 Key/Value。
        aligned_references, _ = self.cross_attention(
            query_tokens,
            reference_tokens,
            reference_tokens,
            key_padding_mask=~reference_valid,
            need_weights=False,
        )
        aligned_references = self.norm(aligned_references)

        features = torch.cat(
            (
                query_tokens,
                aligned_references,
                (query_tokens - aligned_references).abs(),
            ),
            dim=-1,
        )
        # 拼接待测特征、对齐参考特征和二者差异，再恢复成 patch 网格。
        features = features.transpose(1, 2).reshape(
            batch_size, -1, grid_height, grid_width
        )
        pixel_logits = self.head(features)
        pixel_logits = F.interpolate(
            pixel_logits,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )

        # 前景最高 1% patch 的平均分作为整图异常分数。
        foreground_logits = pixel_logits.masked_fill(query_mask < 0.5, -20)
        top_count = height * width // 100
        image_logits = foreground_logits.flatten(1).topk(top_count, dim=1).values.mean(dim=1)
        return pixel_logits, image_logits
