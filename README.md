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
├── reconstruction.py
├── prepare_data.py
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

## 完整 3DGS 前置流程

`reconstruction.py` 保留 can_detect 的完整顺序：正常视频抽帧和 COLMAP、3DGS 训练、SAM2 视频前景传播、按多视角掩码裁剪背景高斯、从清理后的模型插值渲染参考图库。

```bash
python reconstruction.py \
  --video /path/to/normal.mp4 \
  --gaussian-repo /path/to/gaussian-splatting \
  --work-dir /path/to/work/can \
  --model-dir /path/to/models/can \
  --sam2-repo /path/to/sam2 \
  --sam-checkpoint /path/to/sam2.1_hiera_b+.pt \
  --sam-config /path/to/sam2/configs/sam2.1/sam2.1_hiera_b+.yaml \
  --masks-dir masks/can \
  --clean-model-dir /path/to/models/can_clean \
  --references-dir references/can
```

## 重构数据集

附件目录是交付包索引，完整文件位于工作区的 `can-dataset-s43/experiment_data_s43_20261020`。整理脚本只复制原始图像、原始 LabelMe 标注和建库/待测输入清单，并把 13 个易拉罐名称统一成拼音目录名；不会复制现成参考库或任何已生成掩码。

```bash
python prepare_data.py \
  --source /path/to/experiment_data_s43_20261020 \
  --output /path/to/can_detect_raw
```

输出目录可以作为 ModelScope 数据集 `Sondottt/can_detect` 上传；`manifests/reference_inputs.jsonl` 是 3DGS 建库原始输入，`manifests/query_inputs.jsonl` 是全部待测图原始输入，路径均相对于数据集根目录。

`assets/` 只保存 ModelScope 下载代码，不保存权重或数据集。下载脚本中的三个模型分别对应 `can_detect` 的 DINOv2 ViT-S/14、SAM2.1 Hiera B+ 和 Grounding DINO Tiny。
