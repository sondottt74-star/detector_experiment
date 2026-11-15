"""从正常视频训练一个用于生成参考视角的 3DGS 模型。"""

import argparse
import os
from pathlib import Path
import subprocess
import sys


def build_3dgs(video, gaussian_repo, work_dir, model_dir, ffmpeg, colmap, gpu, fps, iterations):
    work_dir = Path(work_dir)
    model_dir = Path(model_dir)
    frames_dir = work_dir / "input"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # 先从正常视频抽帧，再交给 3DGS 官方脚本完成 COLMAP 和训练。
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "warning",
            "-i",
            str(video),
            "-vf",
            f"fps={fps}",
            "-q:v",
            "2",
            str(frames_dir / "frame_%05d.jpg"),
        ],
        check=True,
    )

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
