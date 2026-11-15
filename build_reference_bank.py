"""Build a multi-brand 3DGS reference bank from normal videos.

This orchestrates the existing reconstruction.py pipeline. It intentionally keeps
all generated models, masks, rendered views and the bank index under one output
root so the detector can consume the bank without copying images.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def find_video(video_root: Path, brand: str) -> Path:
    candidates = sorted(video_root.rglob("*.mp4"))
    exact = [p for p in candidates if brand in p.parts]
    if exact:
        return exact[0]
    fuzzy = [p for p in candidates if brand in p.as_posix() or p.parent.parent.name in brand]
    if fuzzy:
        return fuzzy[0]
    raise FileNotFoundError(f"no normal mp4 found for brand {brand!r} under {video_root}")


def build_one(args: argparse.Namespace, repo_root: Path, brand: str, video: Path) -> dict:
    brand_root = Path(args.output_root) / "brands" / brand
    work_dir = brand_root / "work"
    model_dir = brand_root / "model"
    masks_dir = brand_root / "masks"
    clean_model_dir = brand_root / "clean_model"
    references_dir = brand_root / "references"
    references_json = references_dir / "references.json"

    if args.reuse and references_json.exists():
        records = json.loads(references_json.read_text(encoding="utf-8"))
    else:
        command = [
            sys.executable,
            str(repo_root / "reconstruction.py"),
            "--video", str(video),
            "--gaussian-repo", str(args.gaussian_repo),
            "--work-dir", str(work_dir),
            "--model-dir", str(model_dir),
            "--sam2-repo", str(args.sam2_repo),
            "--sam-checkpoint", str(args.sam_checkpoint),
            "--sam-config", str(args.sam_config),
            "--masks-dir", str(masks_dir),
            "--clean-model-dir", str(clean_model_dir),
            "--references-dir", str(references_dir),
            "--reference-count", str(args.reference_count),
            "--fps", str(args.fps),
            "--iterations", str(args.iterations),
            "--gpu", str(args.gpu),
            "--ffmpeg", args.ffmpeg,
            "--colmap", args.colmap,
            "--port", str(args.port),
        ]
        if args.colmap_no_gpu:
            command.append("--colmap-no-gpu")
        print("[reference-bank]", " ".join(command), flush=True)
        environment = os.environ.copy()
        environment["QT_QPA_PLATFORM"] = "offscreen"
        environment["PYTHONPATH"] = ":".join(
            [str(args.gaussian_repo), str(args.sam2_repo)]
            + ([environment["PYTHONPATH"]] if environment.get("PYTHONPATH") else [])
        )
        colmap_path = Path(args.colmap).resolve()
        colmap_root = colmap_path.parent.parent
        plugin_path = colmap_root / "usr" / "lib" / "x86_64-linux-gnu" / "qt5" / "plugins"
        if plugin_path.exists():
            environment["QT_PLUGIN_PATH"] = str(plugin_path)
            environment["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(plugin_path / "platforms")
        library_dirs = [colmap_root / "lib", colmap_root / "usr" / "lib",
                        colmap_root / "usr" / "lib" / "x86_64-linux-gnu",
                        colmap_root / "usr" / "lib" / "x86_64-linux-gnu" / "lapack",
                        colmap_root / "usr" / "lib" / "x86_64-linux-gnu" / "atlas",
                        colmap_root / "usr" / "lib" / "x86_64-linux-gnu" / "blas"]
        existing = environment.get("LD_LIBRARY_PATH", "").split(":")
        environment["LD_LIBRARY_PATH"] = ":".join(
            [str(path) for path in library_dirs if path.exists()] + [item for item in existing if item]
        )
        subprocess.run(command, cwd=repo_root, env=environment, check=True)
        records = json.loads(references_json.read_text(encoding="utf-8"))

    return {
        "brand": brand,
        "video": str(video.resolve()),
        "references_dir": str(references_dir.resolve()),
        "references": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--gaussian-repo", required=True)
    parser.add_argument("--sam2-repo", required=True)
    parser.add_argument("--sam-checkpoint", required=True)
    parser.add_argument("--sam-config", required=True)
    parser.add_argument("--brands", nargs="+", required=True)
    parser.add_argument("--reference-count", type=int, default=720)
    parser.add_argument("--fps", type=float, default=2)
    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--colmap", default="colmap")
    parser.add_argument("--colmap-no-gpu", action="store_true")
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument("--reuse", action="store_true")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    args.gaussian_repo = Path(args.gaussian_repo).resolve()
    args.sam2_repo = Path(args.sam2_repo).resolve()
    args.sam_checkpoint = Path(args.sam_checkpoint).resolve()
    args.sam_config = Path(args.sam_config).resolve()
    for path in (args.gaussian_repo / "convert.py", args.gaussian_repo / "train.py",
                 args.sam2_repo / "sam2" / "build_sam.py", args.sam_checkpoint,
                 args.sam_config):
        if not path.exists():
            raise FileNotFoundError(path)

    video_root = Path(args.video_root).resolve()
    entries = {}
    for brand in args.brands:
        video = find_video(video_root, brand)
        entries[brand] = build_one(args, repo_root, brand, video)

    output = Path(args.output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    bank = {
        "schema_version": 1,
        "reference_count_per_brand": args.reference_count,
        "brands": entries,
    }
    (output / "reference_bank.json").write_text(
        json.dumps(bank, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[reference-bank] wrote {output / 'reference_bank.json'}", flush=True)


if __name__ == "__main__":
    main()
