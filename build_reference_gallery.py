import argparse
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch


def orthogonalize(matrix):
    u, _, vt = np.linalg.svd(matrix)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    return rotation


def interpolate_cameras(cameras, count):
    cameras = sorted(cameras, key=lambda item: item["img_name"])
    total = len(cameras)
    views = []
    for index, value in enumerate(np.linspace(0, total, count, endpoint=False)):
        left_index = int(math.floor(value)) % total
        right_index = (left_index + 1) % total
        alpha = value - math.floor(value)
        left = cameras[left_index]
        right = cameras[right_index]
        left_rotation = np.asarray(left["rotation"], dtype=np.float64).T
        right_rotation = np.asarray(right["rotation"], dtype=np.float64).T
        rotation = orthogonalize(
            (1 - alpha) * left_rotation + alpha * right_rotation
        )
        position = (1 - alpha) * np.asarray(left["position"]) + alpha * np.asarray(
            right["position"]
        )
        views.append(
            {
                "index": index,
                "rotation": rotation,
                "position": position,
                "fx": (1 - alpha) * left["fx"] + alpha * right["fx"],
                "fy": (1 - alpha) * left["fy"] + alpha * right["fy"],
                "width": left["width"],
                "height": left["height"],
                "source_frame": f"{left['img_name']}:{right['img_name']}:{alpha:.6f}",
            }
        )
    return views


def make_camera(view, max_side, MiniCam):
    width = int(view["width"])
    height = int(view["height"])
    scale = min(1, max_side / max(width, height))
    width = round(width * scale)
    height = round(height * scale)
    fx = view["fx"] * width / view["width"]
    fy = view["fy"] * height / view["height"]
    near, far = 0.01, 1000.0
    projection = np.zeros((4, 4), np.float32)
    projection[0, 0] = 2 * fx / width
    projection[1, 1] = 2 * fy / height
    projection[2, 2] = far / (far - near)
    projection[2, 3] = -far * near / (far - near)
    projection[3, 2] = 1
    world = np.eye(4, dtype=np.float32)
    world[:3, :3] = view["rotation"]
    world[:3, 3] = -view["rotation"] @ view["position"]
    world = torch.tensor(world, device="cuda").T.contiguous()
    projection = torch.tensor(projection, device="cuda").T.contiguous()
    camera = MiniCam(
        width,
        height,
        2 * math.atan(height / (2 * fy)),
        2 * math.atan(width / (2 * fx)),
        near,
        far,
        world,
        world @ projection,
    )
    return camera, width, height


def build_gallery(gaussian_repo, model_dir, output_dir, views, max_side, iteration):
    sys.path.insert(0, str(gaussian_repo))
    from gaussian_renderer import render
    from scene import GaussianModel
    from scene.cameras import MiniCam

    model_dir = Path(model_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cameras = json.loads((model_dir / "cameras.json").read_text())
    views = interpolate_cameras(cameras, views)
    gaussian = GaussianModel(3)
    gaussian.load_ply(
        str(model_dir / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply")
    )
    pipeline = SimpleNamespace(
        convert_SHs_python=False,
        compute_cov3D_python=False,
        debug=False,
        antialiasing=False,
    )
    background = torch.zeros(3, device="cuda")
    white = torch.ones((gaussian.get_xyz.shape[0], 3), device="cuda")
    records = []

    with torch.inference_mode():
        for view in views:
            camera, width, height = make_camera(view, max_side, MiniCam)
            rgb = render(camera, gaussian, pipeline, background)["render"].clamp(0, 1)
            silhouette = render(
                camera,
                gaussian,
                pipeline,
                background,
                override_color=white,
            )["render"][0].clamp(0, 1)
            foreground = silhouette.cpu().numpy() > 0.5
            pixels = (rgb.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            pixels[~foreground] = 0
            image_path = output_dir / f"{view['index']:04d}.jpg"
            mask_path = output_dir / f"{view['index']:04d}_mask.png"
            Image.fromarray(pixels).save(image_path, quality=94, subsampling=0)
            Image.fromarray(foreground.astype(np.uint8) * 255).save(mask_path)
            records.append(
                {
                    "image": str(image_path.resolve()),
                    "foreground_mask": str(mask_path.resolve()),
                    "source_frame": view["source_frame"],
                    "reference_index": view["index"],
                }
            )

    (output_dir / "references.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gaussian-repo", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--views", type=int, default=720)
    parser.add_argument("--max-side", type=int, default=720)
    parser.add_argument("--iteration", type=int, default=30000)
    args = parser.parse_args()
    build_gallery(
        args.gaussian_repo,
        args.model_dir,
        args.output_dir,
        args.views,
        args.max_side,
        args.iteration,
    )
