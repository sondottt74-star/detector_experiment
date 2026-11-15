import argparse
from pathlib import Path

from modelscope import snapshot_download


MODEL_IDS = {
    "dinov2": "facebook/dinov2-small",
    "sam2": "facebook/sam2.1-hiera-base-plus",
    "grounding_dino": "muse/grounding-dino-base",
}


def download_model(model_id, output_dir):
    return snapshot_download(
        model_id=model_id,
        local_dir=str(output_dir),
    )


def download_project_models(output_root, names):
    output_root = Path(output_root)
    for name in names:
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
