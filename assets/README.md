# ModelScope assets

这里仅保存下载代码，不保存任何模型权重或数据集文件。

下载接口对应 ModelScope 的模型下载和数据集下载 SDK。

## 下载项目预训练模型

```bash
python assets/download_modelscope_models.py \
  --output-root assets/pretrained
```

只下载其中一个：

```bash
python assets/download_modelscope_models.py \
  --model dinov2_vits14 \
  --output-root assets/pretrained
```

当前代码中的 ModelScope 模型 ID：

```text
dinov2_vits14 → facebook/dinov2-small
sam2.1_hiera_b_plus → facebook/sam2.1-hiera-base-plus
```

`can_detect` 使用的 Grounding DINO 是 `grounding-dino-tiny`。目前没有确认到对应的 ModelScope Tiny 仓库，因此这里不把 `muse/grounding-dino-base` 当作替代模型。

## 下载 MVTec AD

ModelScope 的数据集下载接口需要具体的 `namespace/name`。把实际的 MVTec AD ModelScope 仓库 ID 传入：

```bash
python assets/download_mvtec_ad.py \
  --dataset-id YOUR_NAMESPACE/YOUR_MVTEC_AD_DATASET \
  --output-dir assets/datasets/mvtec_ad
```

脚本只调用 ModelScope 的 `dataset_snapshot_download`，不会在仓库中保存下载结果。
