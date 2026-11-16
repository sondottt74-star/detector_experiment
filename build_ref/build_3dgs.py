"""从正常视频训练一个用于生成参考视角的 3DGS 模型。"""

import argparse
import os
import shutil
from pathlib import Path
import subprocess
import sys



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

def build_3dgs(video, gaussian_repo, work_dir, model_dir, ffmpeg, colmap, gpu, fps, iterations):
    work_dir = Path(work_dir)
    model_dir = Path(model_dir)
    frames_dir = work_dir / "input"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # 先从正常视频抽帧，再交给 3DGS 官方脚本完成 COLMAP 和训练。
    extract_video_frames(video, frames_dir, fps, ffmpeg)

    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    subprocess.run(
        [
            sys.executable,
            "convert.py",
            "-s",
            str(work_dir),
            "--colmap_executable",
            str(colmap),
        ],
        cwd=gaussian_repo,
        env=environment,
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "train.py",
            "-s",
            str(work_dir),
            "-m",
            str(model_dir),
            "--iterations",
            str(iterations),
            "--save_iterations",
            str(iterations),
        ],
        cwd=gaussian_repo,
        env=environment,
        check=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--gaussian-repo", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--colmap", default="colmap")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--fps", type=float, default=2)
    parser.add_argument("--iterations", type=int, default=30000)
    args = parser.parse_args()
    build_3dgs(
        args.video,
        args.gaussian_repo,
        args.work_dir,
        args.model_dir,
        args.ffmpeg,
        args.colmap,
        args.gpu,
        args.fps,
        args.iterations,
    )
