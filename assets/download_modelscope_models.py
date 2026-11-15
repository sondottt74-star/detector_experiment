"""从 ModelScope 获取与 can_detect 对应的预训练模型。"""

import argparse
from pathlib import Path

from modelscope import snapshot_download


MODEL_IDS = {
    # can_detect 使用 official DINOv2 ViT-S/14，ModelScope 对应 small 模型。
    "dinov2_vits14": "facebook/dinov2-small",
    # can_detect 的配置是 sam2.1_hiera_b+。
    "sam2.1_hiera_b_plus": "facebook/sam2.1-hiera-base-plus",
}


def download_model(model_id, output_dir):
    # ModelScope 会把模型文件写到指定的本地目录。
    return snapshot_download(
        model_id=model_id,
        local_dir=str(output_dir),
    )


def download_project_models(output_root, names):
    output_root = Path(output_root)
    for name in names:
        # 每个模型单独放置，便于后续在配置中填写路径。
        download_model(MODEL_IDS[name], output_root / name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="assets/pretrained")
    parser.add_argument("--model", choices=MODEL_IDS, action="append")
    args = parser.parse_args()
    download_project_models(
        args.output_root,
        args.model or list(MODEL_IDS),
    )
