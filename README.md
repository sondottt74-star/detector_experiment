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
├── build_3dgs.py
├── build_reference_gallery.py
├── segment_foreground.py
├── assets/
├── references/
├── masks/
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

## build inputs

从正常视频训练本地 3DGS：

```bash
python build_3dgs.py \
  --video /path/to/normal.mp4 \
  --gaussian-repo /path/to/gaussian-splatting \
  --work-dir /path/to/work/can \
  --model-dir /path/to/models/can
```

从 3DGS 模型渲染正常参考图库：

```bash
python build_reference_gallery.py \
  --gaussian-repo /path/to/gaussian-splatting \
  --model-dir /path/to/models/can \
  --output-dir references/can \
  --views 720
```

对待测图提取 SAM2 前景：

```bash
python segment_foreground.py \
  --input-dir /path/to/query_images \
  --output-dir masks/query \
  --sam2-repo /path/to/sam2 \
  --checkpoint /path/to/sam2_checkpoint.pt \
  --config /path/to/sam2/configs/sam2.1/sam2.1_hiera_b+.yaml
```

`build_reference_gallery.py` 输出参考图、参考掩码和 `references.json`；
`segment_foreground.py` 输出待测图前景掩码和 `foreground.jsonl`。两个脚本都只读取本地仓库和本地权重。

`assets/` 只保存 ModelScope 下载代码，不保存权重或数据集。下载脚本中的三个模型分别对应 `can_detect` 的 DINOv2 ViT-S/14、SAM2.1 Hiera B+ 和 Grounding DINO Tiny。
