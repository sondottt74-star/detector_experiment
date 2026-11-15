"""3DGS 参考库的完整实验流程：重建、SAM2 分割、裁剪高斯、渲染参考图。"""

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path



def extract_video_frames(video, frames_dir, fps, ffmpeg):
    """Extract frames with local ffmpeg, falling back to OpenCV."""
    frames_dir = Path(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg_path = shutil.which(str(ffmpeg)) or (str(ffmpeg) if Path(ffmpeg).exists() else None)
    if ffmpeg_path:
        subprocess.run([ffmpeg_path, "-hide_banner", "-loglevel", "warning", "-i",
                        str(video), "-vf", f"fps={fps}", "-q:v", "2",
                        str(frames_dir / "frame_%05d.jpg")], check=True)
        return
    import cv2
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {video}")
    source_fps = capture.get(cv2.CAP_PROP_FPS) or fps
    stride = max(1, round(source_fps / fps))
    frame_index = 0
    output_index = 1
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index % stride == 0:
            target = frames_dir / f"frame_{output_index:05d}.jpg"
            if not cv2.imwrite(str(target), frame):
                raise RuntimeError(f"cannot write frame: {target}")
            output_index += 1
        frame_index += 1
    capture.release()
    if output_index == 1:
        raise RuntimeError(f"no frames extracted from video: {video}")

def run_3dgs(video, gaussian_repo, work_dir, model_dir, sam2_repo, sam_checkpoint,
             sam_config, masks_dir, clean_model_dir, references_dir, reference_count,
             ffmpeg, colmap, gpu, fps, iterations, colmap_no_gpu=False, port=6009):
    # 1. 抽帧并用 COLMAP 建立相机，2. 训练正常 3DGS。
    work_dir = Path(work_dir)
    model_dir = Path(model_dir)
    frames_dir = work_dir / "input"
    frames_dir.mkdir(parents=True, exist_ok=True)
    extract_video_frames(video, frames_dir, fps, ffmpeg)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    convert_command = [sys.executable, "convert.py", "-s", str(work_dir),
                       "--colmap_executable", str(colmap)]
    if colmap_no_gpu:
        convert_command.append("--no_gpu")
    subprocess.run(convert_command, cwd=gaussian_repo, env=environment, check=True)
    subprocess.run([sys.executable, "train.py", "-s", str(work_dir), "-m", str(model_dir),
                    "--iterations", str(iterations), "--save_iterations", str(iterations),
                    "--ip", "127.0.0.1", "--port", str(port)],
                   cwd=gaussian_repo, env=environment, check=True)
    # 3. SAM2 在重建视频上做前景传播，4. 用多视角掩码裁掉背景高斯。
    segment_model(frames_dir, masks_dir, sam2_repo, sam_checkpoint, sam_config)
    prune_model(model_dir, masks_dir, clean_model_dir, iterations)
    # 5. 从清理后的模型插值渲染正常参考图库。
    render_reference(gaussian_repo, clean_model_dir, references_dir, reference_count, iterations)


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


def clean_mask(mask):
    import cv2
    import numpy as np
    mask = mask.astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count > 1:
        mask = labels == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    flood = mask.copy()
    holder = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), np.uint8)
    cv2.floodFill(flood, holder, (0, 0), 1)
    return (mask | (1 - flood)).astype(bool)


def segment_model(frames_dir, masks_dir, sam2_repo, checkpoint, config):
    import numpy as np
    import torch
    from PIL import Image

    sys.path.insert(0, str(sam2_repo))
    from sam2.build_sam import build_sam2_video_predictor

    # 第一帧使用框和正负点提示，之后由 SAM2 视频预测器传播到全部帧。
    frames = sorted(Path(frames_dir).glob("*.jpg"))
    output = Path(masks_dir)
    output.mkdir(parents=True, exist_ok=True)
    predictor = build_sam2_video_predictor(
        _config_name(config, sam2_repo), str(checkpoint), device="cuda"
    )
    with tempfile.TemporaryDirectory(prefix="can_frames_") as staging:
        staging = Path(staging)
        for index, frame in enumerate(frames):
            (staging / f"{index:06d}.jpg").symlink_to(frame.resolve())
        state = predictor.init_state(video_path=str(staging), offload_video_to_cpu=True,
                                     offload_state_to_cpu=False, async_loading_frames=False)
        height, width = state["video_height"], state["video_width"]
        box = np.array([.34 * width, .22 * height, .66 * width, .91 * height], np.float32)
        points = np.array([[.50 * width, .43 * height], [.50 * width, .55 * height],
                           [.50 * width, .68 * height], [.20 * width, .55 * height],
                           [.80 * width, .55 * height], [.50 * width, .96 * height]], np.float32)
        labels = np.array([1, 1, 1, 0, 0, 0], np.int32)
        predictor.add_new_points_or_box(inference_state=state, frame_idx=0, obj_id=1,
                                        points=points, labels=labels, box=box)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for frame_idx, object_ids, logits in predictor.propagate_in_video(state):
                object_index = list(object_ids).index(1)
                mask = clean_mask((logits[object_index, 0] > 0).cpu().numpy())
                Image.fromarray(mask.astype(np.uint8) * 255).save(
                    output / f"{frames[frame_idx].stem}.png"
                )
        predictor.reset_state(state)


def load_mask(path, width, height, dilation):
    import cv2
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    mask = mask > 127
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilation + 1,) * 2)
    return cv2.dilate(mask.astype("uint8"), kernel) > 0


def prune_model(model_dir, masks_dir, clean_model_dir, iteration, min_ratio=.55, min_views=10):
    import numpy as np
    from plyfile import PlyData, PlyElement

    # 每个高斯投影到所有相机，只有在足够多的前景掩码中出现才保留。
    source = Path(model_dir)
    destination = Path(clean_model_dir)
    source_ply = source / f"point_cloud/iteration_{iteration}/point_cloud.ply"
    cameras = json.loads((source / "cameras.json").read_text(encoding="utf-8"))
    ply = PlyData.read(str(source_ply))
    vertices = ply["vertex"].data
    xyz = np.column_stack((vertices["x"], vertices["y"], vertices["z"])).astype(np.float64)
    inside = np.zeros(len(xyz), dtype=np.uint16)
    used = 0
    for camera in cameras:
        mask_path = Path(masks_dir) / f"{Path(camera['img_name']).stem}.png"
        mask = load_mask(mask_path, int(camera["width"]), int(camera["height"]), 4)
        width, height = int(camera["width"]), int(camera["height"])
        rotation = np.asarray(camera["rotation"], dtype=np.float64)
        center = np.asarray(camera["position"], dtype=np.float64)
        fx, fy = float(camera["fx"]), float(camera["fy"])
        used += 1
        for start in range(0, len(xyz), 150000):
            end = min(start + 150000, len(xyz))
            camera_xyz = (xyz[start:end] - center) @ rotation
            z = camera_xyz[:, 2]
            valid_z = z > 1e-6
            u = np.zeros_like(z)
            v = np.zeros_like(z)
            u[valid_z] = fx * camera_xyz[valid_z, 0] / z[valid_z] + width * .5
            v[valid_z] = fy * camera_xyz[valid_z, 1] / z[valid_z] + height * .5
            ui = np.floor(u + .5).astype(np.int64)
            vi = np.floor(v + .5).astype(np.int64)
            valid = valid_z & (ui >= 0) & (ui < width) & (vi >= 0) & (vi < height)
            ids = np.flatnonzero(valid)
            inside[start:end][ids[mask[vi[ids], ui[ids]]]] += 1
    required = max(min_views, math.ceil(min_ratio * used))
    keep = inside >= required
    destination_ply = destination / f"point_cloud/iteration_{iteration}/point_cloud.ply"
    destination_ply.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(vertices[keep], "vertex")], text=ply.text,
            byte_order=ply.byte_order, comments=list(ply.comments),
            obj_info=list(ply.obj_info)).write(str(destination_ply))
    for filename in ("cfg_args", "cameras.json", "exposure.json"):
        if (source / filename).exists():
            shutil.copy2(source / filename, destination / filename)


def render_reference(gaussian_repo, model_dir, references_dir, count, iteration):
    import numpy as np
    import torch
    from PIL import Image
    from types import SimpleNamespace

    sys.path.insert(0, str(gaussian_repo))
    from gaussian_renderer import render
    from scene import GaussianModel
    from scene.cameras import MiniCam
    from build_reference_gallery import interpolate_cameras, make_camera

    # 清理后的高斯模型沿相机轨迹插值，输出 RGB 参考图和前景掩码。
    model = Path(model_dir)
    output = Path(references_dir)
    output.mkdir(parents=True, exist_ok=True)
    cameras = json.loads((model / "cameras.json").read_text(encoding="utf-8"))
    views = interpolate_cameras(cameras, count)
    gaussian = GaussianModel(3)
    gaussian.load_ply(str(model / f"point_cloud/iteration_{iteration}/point_cloud.ply"))
    pipeline = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False,
                               debug=False, antialiasing=False)
    background = torch.zeros(3, device="cuda")
    white = torch.ones((gaussian.get_xyz.shape[0], 3), device="cuda")
    records = []
    with torch.inference_mode():
        for view in views:
            camera, width, height = make_camera(view, 720, MiniCam)
            rgb = render(camera, gaussian, pipeline, background)["render"].clamp(0, 1)
            silhouette = render(camera, gaussian, pipeline, background,
                                override_color=white)["render"][0].clamp(0, 1)
            foreground = silhouette.cpu().numpy() > .5
            pixels = (rgb.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            pixels[~foreground] = 0
            image_path = output / f"{view['index']:04d}.jpg"
            mask_path = output / f"{view['index']:04d}_mask.png"
            Image.fromarray(pixels).save(image_path, quality=94, subsampling=0)
            Image.fromarray(foreground.astype(np.uint8) * 255).save(mask_path)
            records.append({"image": str(image_path.resolve()),
                            "foreground_mask": str(mask_path.resolve()),
                            "source_frame": view["source_frame"],
                            "reference_index": view["index"]})
    (output / "references.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def update_bank(references_root, bank_path, names):
    bank = json.loads(Path(bank_path).read_text(encoding="utf-8")) if Path(bank_path).exists() else {}
    for name in names:
        bank[name] = json.loads((Path(references_root) / name / "references.json").read_text())
    Path(bank_path).write_text(json.dumps(bank, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--gaussian-repo", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--sam2-repo", required=True)
    parser.add_argument("--sam-checkpoint", required=True)
    parser.add_argument("--sam-config", required=True)
    parser.add_argument("--masks-dir", required=True)
    parser.add_argument("--clean-model-dir", required=True)
    parser.add_argument("--references-dir", required=True)
    parser.add_argument("--reference-count", type=int, default=720)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--colmap", default="colmap")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--fps", type=float, default=2)
    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument("--colmap-no-gpu", action="store_true")
    parser.add_argument("--port", type=int, default=6009)
    args = parser.parse_args()
    run_3dgs(args.video, args.gaussian_repo, args.work_dir, args.model_dir,
              args.sam2_repo, args.sam_checkpoint, args.sam_config, args.masks_dir,
              args.clean_model_dir, args.references_dir, args.reference_count,
              args.ffmpeg, args.colmap, args.gpu, args.fps, args.iterations,
              args.colmap_no_gpu, args.port)
