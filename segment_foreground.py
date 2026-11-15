"""用本地 SAM2 从待测图中提取易拉罐前景。"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch


def _config_name(config, sam2_repo):
    """Convert a local SAM2 config path to Hydra's package-relative name."""
    config = Path(config)
    sam2_repo = Path(sam2_repo)
    if config.is_absolute():
        try:
            config = config.relative_to(sam2_repo / "sam2")
        except ValueError:
            pass
    return config.as_posix()


def segment_directory(input_dir, output_dir, sam2_repo, checkpoint, config, box):
    sys.path.insert(0, str(sam2_repo))
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictor = SAM2ImagePredictor(
        build_sam2(_config_name(config, sam2_repo), str(checkpoint), device="cuda")
    )
    records = []

    image_paths = sorted(
        path for path in input_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )
    for image_path in image_paths:
        # 用固定比例框和罐体内部正点提示 SAM2。
        image = Image.open(image_path).convert("RGB")
        image_array = np.asarray(image)
        height, width = image_array.shape[:2]
        image_box = np.asarray(
            [
                box[0] * width,
                box[1] * height,
                box[2] * width,
                box[3] * height,
            ],
            dtype=np.float32,
        )
        points = np.asarray(
            [
                [(image_box[0] + image_box[2]) / 2, image_box[1] + r * (image_box[3] - image_box[1])]
                for r in (0.3, 0.5, 0.7)
            ],
            dtype=np.float32,
        )
        predictor.set_image(image_array)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            masks, scores, _ = predictor.predict(
                box=image_box,
                point_coords=points,
                point_labels=np.ones(3, dtype=np.int32),
                multimask_output=True,
            )
        mask = masks[scores.argmax()]
        mask_path = output_dir / f"{image_path.stem}_mask.png"
        Image.fromarray(mask.astype(np.uint8) * 255).save(mask_path)
        records.append(
            {
                "image": str(image_path.resolve()),
                "foreground_mask": str(mask_path.resolve()),
            }
        )

    (output_dir / "foreground.jsonl").write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sam2-repo", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--box",
        type=float,
        nargs=4,
        default=(0.2, 0.1, 0.8, 0.95),
        metavar=("X0", "Y0", "X1", "Y1"),
    )
    args = parser.parse_args()
    segment_directory(
        args.input_dir,
        args.output_dir,
        args.sam2_repo,
        args.checkpoint,
        args.config,
        args.box,
    )
