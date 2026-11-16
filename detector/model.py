"""参考图条件异常检测器：DINOv2、交叉注意力和像素级检测头。"""

import torch
from torch import nn
from torch.nn import functional as F


class Detector(nn.Module):
    """Frozen DINOv2, reference cross attention and anomaly prediction head.

    待测图的每个 patch 到正常参考图里对齐外观，再和参考特征做差，
    输出像素热力图和整图异常分数。编码器冻结，只训练后面的层。
    """

    def __init__(self, encoder, feature_dim=384, hidden_dim=128):
        super().__init__()
        # feature_dim=384 对应 DINOv2 ViT-S/14。编码器只提特征，不更新。
        self.encoder = encoder.requires_grad_(False).eval()
        self.project = nn.Linear(feature_dim, hidden_dim)
        # 待测 patch 作 Query，参考 patch 作 Key/Value。
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, num_heads=4, batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)
        # 三路拼接：待测特征、对齐后的参考、二者绝对差，通道数是 hidden_dim * 3。
        self.head = nn.Sequential(
            nn.Conv2d(hidden_dim * 3, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 1, 1),
        )

    def train(self, mode=True):
        # 外层切到 train 时仍强制编码器保持 eval，避免 Dropout 和统计量更新。
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
        # ViT-S/14 的 patch 边长是 14。518 的输入对应 37×37 个 patch。
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

        # 前景掩码下采样到 patch 网格。True 表示该参考 patch 可以参与对齐。
        reference_valid = F.adaptive_max_pool2d(
            reference_masks.reshape(batch_size * reference_count, 1, height, width),
            (grid_height, grid_width),
        )
        reference_valid = reference_valid.reshape(
            batch_size, reference_count * token_count
        ) > 0

        # Query 是待测图，Key/Value 是正常参考图。
        # key_padding_mask 中 True 表示忽略，所以对有效前景取反。
        aligned_references, _ = self.cross_attention(
            query_tokens,
            reference_tokens,
            reference_tokens,
            key_padding_mask=~reference_valid,
            need_weights=False,
        )
        aligned_references = self.norm(aligned_references)

        # 绝对差把外观偏离直接交给检测头，不必只靠网络自己学比较。
        features = torch.cat(
            (
                query_tokens,
                aligned_references,
                (query_tokens - aligned_references).abs(),
            ),
            dim=-1,
        )
        features = features.transpose(1, 2).reshape(
            batch_size, -1, grid_height, grid_width
        )
        pixel_logits = self.head(features)
        # 检测头在 patch 网格上预测，再插值回原图分辨率。
        pixel_logits = F.interpolate(
            pixel_logits,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )

        # 背景填一个很小的 logit，避免进入 top-k。
        # 前景最高 1% 像素的平均分作为整图异常分数，对应局部缺陷。
        foreground_logits = pixel_logits.masked_fill(query_mask < 0.5, -20)
        top_count = height * width // 100
        image_logits = foreground_logits.flatten(1).topk(top_count, dim=1).values.mean(dim=1)
        return pixel_logits, image_logits
