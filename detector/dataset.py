"""读取待测图、正常参考图和对应掩码。"""

import torch
from torch.utils.data import Dataset

from .utils import load_mask, load_rgb, read_jsonl


class ReferenceDataset(Dataset):
    def __init__(self, manifest, image_size=518):
        self.records = read_jsonl(manifest)
        # 尺寸在写清单时已经固定，这里保留参数，读取时不再缩放。
        self.image_size = image_size

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        # 清单里的图和掩码在划分数据集时已经做成 image_size 的正方形。
        query = load_rgb(record["query"])
        references = torch.stack([load_rgb(path) for path in record["references"]])
        query_mask = load_mask(record["query_mask"])
        reference_masks = torch.stack([load_mask(path) for path in record["reference_masks"]])
        target_mask = load_mask(record["target_mask"])
        label = torch.tensor(record["label"], dtype=torch.float32)
        return query, references, query_mask, reference_masks, target_mask, label
