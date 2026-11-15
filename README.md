# Detector experiment

这是一个最小实验目录，使用本地 DINOv2 完成：

```text
query + normal references
→ DINOv2 patch tokens
→ reference cross attention
→ query/reference/difference features
→ pixel anomaly heatmap + image anomaly score
```

## 目录

```text
detector_experiment/
├── model.py
├── dataset.py
├── trainer.py
├── utils.py
├── main.py
├── output/
├── save/
└── log/
```

## manifest

每行一个 JSON：

```json
{
  "query": "query.png",
  "references": ["ref_1.png", "ref_2.png", "ref_3.png"],
  "query_mask": "query_mask.png",
  "reference_masks": ["ref_1_mask.png", "ref_2_mask.png", "ref_3_mask.png"],
  "target_mask": "target_mask.png",
  "label": 1
}
```

## run

DINOv2 仓库和权重都从本地读取，不下载 PyTorch 或模型：

```bash
cd detector_experiment
python main.py \
  --train-manifest /path/to/train.jsonl \
  --val-manifest /path/to/val.jsonl \
  --dino-repo /path/to/dinov2/repo \
  --dino-weights /path/to/dinov2_vits14_pretrain.pth
```
