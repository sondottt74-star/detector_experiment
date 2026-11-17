"""只覆盖 target.png。查询图、SAM 掩码和参考图保持不动。"""

import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from detector.utils import fit_foreground
from preprocessing.build_detection_manifest import write_target_mask
from preprocessing.prepare_queries import cache_key, defect_boxes

# 清单在 /home/zhanglesong/foldN/manifest.json，相对路径从它的上两级拼。
DATA_ROOT = Path("/home/zhanglesong")
MANIFESTS = [DATA_ROOT / f"fold{fold}/manifest.json" for fold in range(1, 6)]
MANIFESTS.append(DATA_ROOT / "zeroshot/manifest.json")
# 当前训练清单指向这份缓存。prepare_queries.py 里的常量还写着 prepared。
QUERY_CACHE = ROOT / "cleaned_data" / "_shared" / "queries"
SIZE = 518


def locate(value):
    path = Path(value)
    if path.is_absolute():
        return path
    return DATA_ROOT / path


def query_records():
    """六份清单里的待测图按原图路径去重。建库抽帧没有 target.png。"""
    seen = set()
    for manifest in MANIFESTS:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        for record in payload["records"]:
            if record.get("role") == "reference_video_frame":
                continue
            image_path = locate(record["image"]).resolve()
            if image_path in seen:
                continue
            seen.add(image_path)
            yield record, image_path


def redraw(record, image_path):
    cached = QUERY_CACHE / cache_key(image_path)
    sam_path = cached / "sam.png"
    target_path = cached / "target.png"
    if not sam_path.is_file():
        raise FileNotFoundError(f"missing sam mask for {image_path}")
    with Image.open(image_path) as image, Image.open(sam_path) as sam_mask:
        _, _, window = fit_foreground(image, sam_mask, SIZE)
    annotation = record.get("annotation")
    boxes = defect_boxes(locate(annotation), image_path) if annotation else []
    write_target_mask(boxes, window, target_path, SIZE)
    return len(boxes), target_path


def source_by_stem():
    """原图文件名对应标注。fold2 和 fold3 的副本用这个对回去。"""
    found = {}
    for record, image_path in query_records():
        found[(record.get("brand"), image_path.stem)] = (record, image_path)
    return found


def redraw_local_copies():
    """fold2、fold3 的 train 和 val 指向各自目录里的 target，不在共享缓存。"""
    sources = source_by_stem()
    rewritten = 0
    missing = []
    for fold in (2, 3):
        for target_path in (ROOT / "cleaned_data" / f"fold{fold}" / "queries").rglob("*_target.png"):
            prefix = target_path.name[: -len("_target.png")]
            stem = prefix.split("_", 1)[1]
            sam_path = target_path.with_name(prefix + "_sam.png")
            key = (target_path.parent.name, stem)
            if key not in sources or not sam_path.is_file():
                missing.append(str(target_path))
                continue
            record, image_path = sources[key]
            with Image.open(image_path) as image, Image.open(sam_path) as sam_mask:
                _, _, window = fit_foreground(image, sam_mask, SIZE)
            annotation = record.get("annotation")
            boxes = defect_boxes(locate(annotation), image_path) if annotation else []
            write_target_mask(boxes, window, target_path, SIZE)
            rewritten += 1
    if missing:
        raise FileNotFoundError(f"unmatched local targets: {missing[:5]} n={len(missing)}")
    print(f"rewrote local copies {rewritten}", flush=True)


def main():
    rewritten = 0
    with_boxes = 0
    for record, image_path in query_records():
        count, _ = redraw(record, image_path)
        rewritten += 1
        with_boxes += int(count > 0)
        if rewritten % 200 == 0:
            print(f"rewrote {rewritten}", flush=True)
    print(f"rewrote {rewritten} with_defect_boxes={with_boxes}", flush=True)
    redraw_local_copies()


if __name__ == "__main__":
    main()
