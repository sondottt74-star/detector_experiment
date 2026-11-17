"""读取待测图、正常参考图和对应掩码。"""

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from .utils import _MEAN_U8, load_mask, load_rgb, read_jsonl, rgb_tensor


def foundad_appearance(image_size):
    """FoundAD 外观增强：以 0.3 的概率在颜色抖动、灰度和模糊里只选一种。"""
    kernel = 23 if image_size >= 384 else 11
    return transforms.RandomApply(
        [
            transforms.RandomChoice(
                [
                    transforms.ColorJitter(0.3, 0.3, 0.3, 0.05),
                    transforms.RandomGrayscale(p=1.0),
                    transforms.GaussianBlur(kernel_size=kernel, sigma=(0.1, 2.0)),
                ]
            )
        ],
        p=0.3,
    )


class ReferenceDataset(Dataset):
    def __init__(self, manifest, image_size=518, augment=False, reference_count=4):
        self.records = read_jsonl(manifest)
        # 尺寸在写清单时已经固定，这里保留参数，读取时不再缩放。
        self.image_size = image_size
        self.augment = foundad_appearance(image_size) if augment else None
        # 清单按相似度从高到低排列，只取前 reference_count 张。
        self.reference_count = reference_count

    def __len__(self):
        return len(self.records)

    def _load_rgb(self, path, mask_path):
        if self.augment is None:
            return load_rgb(path)
        image = self.augment(Image.open(path).convert("RGB"))
        mask = np.asarray(Image.open(mask_path).convert("L")) > 127
        array = np.asarray(image).copy()
        array[~mask] = _MEAN_U8
        return rgb_tensor(Image.fromarray(array))

    def __getitem__(self, index):
        record = self.records[index]
        # 清单里的图和掩码在划分数据集时已经做成 image_size 的正方形。
        # query 和每张参考图各自采样一次，掩码本身不改。
        query = self._load_rgb(record["query"], record["query_mask"])
        reference_paths = record["references"][: self.reference_count]
        reference_mask_paths = record["reference_masks"][: self.reference_count]
        references = torch.stack(
            [
                self._load_rgb(path, mask)
                for path, mask in zip(reference_paths, reference_mask_paths)
            ]
        )
        query_mask = load_mask(record["query_mask"])
        reference_masks = torch.stack([load_mask(path) for path in reference_mask_paths])
        target_mask = load_mask(record["target_mask"])
        label = torch.tensor(record["label"], dtype=torch.float32)
        return query, references, query_mask, reference_masks, target_mask, label
