"""读取待测图、正常参考图和对应掩码。"""

import torch
from torch.utils.data import Dataset

from utils import load_mask, load_rgb, read_jsonl


class ReferenceDataset(Dataset):
    def __init__(self, manifest, image_size=448):
        self.records = read_jsonl(manifest)
        self.image_size = image_size

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        # manifest 中的 references 与 reference_masks 按同一顺序配对。
        query = load_rgb(record["query"], self.image_size)
        references = torch.stack(
            [load_rgb(path, self.image_size) for path in record["references"]]
        )
        query_mask = load_mask(record["query_mask"], self.image_size)
        reference_masks = torch.stack(
            [load_mask(path, self.image_size) for path in record["reference_masks"]]
        )
        target_mask = load_mask(record["target_mask"], self.image_size)
        label = torch.tensor(record["label"], dtype=torch.float32)
        return query, references, query_mask, reference_masks, target_mask, label
