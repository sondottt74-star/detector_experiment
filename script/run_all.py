"""在空闲 GPU 上排队跑五折监督和 zeroshot，每张卡同时只跑一份。"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.train_config import resolve


CONFIGS = [
    "config/photometric/fold1.yaml",
    "config/photometric/fold2.yaml",
    "config/photometric/fold3.yaml",
    "config/photometric/fold4.yaml",
    "config/photometric/fold5.yaml",
    "config/photometric/zeroshot.yaml",
]
PREFERRED_GPUS = ["1", "3"]


def idle_gpus(candidates):
    text = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    free = []
    busy = []
    for line in text.strip().splitlines():
        index, memory, util = [part.strip() for part in line.split(",")]
        if index not in candidates:
            continue
        # 显存和占用都低，才当作这张卡可以独占。
        if int(float(memory)) < 8000 and int(float(util)) < 30:
            free.append(index)
        else:
            busy.append(f"gpu {index}: {memory} MiB, util {util}%")
    return free, busy


def metrics_path(config_path):
    raw = yaml.safe_load((ROOT / config_path).read_text(encoding="utf-8"))
    return resolve(raw["infer_output"]) / "metrics.json"


def summarize(finished):
    rows = []
    for config_path, return_code in finished:
        item = {"config": config_path, "return_code": return_code}
        path = metrics_path(config_path)
        if path.is_file():
            item["metrics"] = json.loads(path.read_text(encoding="utf-8"))
        rows.append(item)
    supervised = [
        row for row in rows
        if "zeroshot" not in row["config"] and "metrics" in row
    ]
    summary = {"runs": rows}
    if supervised:
        keys = ("image_auroc", "image_ap", "pixel_auroc", "pixel_ap", "f1_at_0.5")
        mean = {}
        for key in keys:
            values = [
                row["metrics"][key] for row in supervised
                if row["metrics"].get(key) is not None
            ]
            mean[key] = sum(values) / len(values) if values else None
            mean[f"n_{key}"] = len(values)
        summary["supervised_mean"] = mean
    destination = ROOT / "log" / "summary.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"summary -> {destination}", flush=True)
    for row in rows:
        metrics = row.get("metrics") or {}
        print(
            f"{row['config']} code={row['return_code']} "
            f"image_auroc={metrics.get('image_auroc')} "
            f"image_ap={metrics.get('image_ap')} "
            f"pixel_auroc={metrics.get('pixel_auroc')}",
            flush=True,
        )
    if "supervised_mean" in summary:
        print(f"supervised_mean={summary['supervised_mean']}", flush=True)
    return summary


def start_job(python, gpu, config_path):
    log_path = ROOT / "log" / Path(config_path).with_suffix("").name
    # fold 配置同名会撞车，日志跟配置目录走。
    raw = yaml.safe_load((ROOT / config_path).read_text(encoding="utf-8"))
    log_path = resolve(raw["log_dir"]) / "run.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("w", encoding="utf-8")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["PYTHONUNBUFFERED"] = "1"
    env["OMP_NUM_THREADS"] = "16"
    env["MKL_NUM_THREADS"] = "16"
    process = subprocess.Popen(
        [python, str(ROOT / "script" / "run_one.py"), str(ROOT / config_path)],
        cwd=ROOT,
        env=env,
        stdout=handle,
        stderr=subprocess.STDOUT,
    )
    print(f"start gpu={gpu} pid={process.pid} {config_path} log={log_path}", flush=True)
    return process, handle, log_path


def main():
    python = str(ROOT / ".venv" / "bin" / "python")
    requested = PREFERRED_GPUS
    if len(sys.argv) == 2:
        requested = [item.strip() for item in sys.argv[1].split(",") if item.strip()]
    gpus, busy = idle_gpus(requested)
    for line in busy:
        print(f"skip busy {line}", flush=True)
    if not gpus:
        raise SystemExit("no idle gpu among " + ",".join(requested))
    print(f"using gpus={','.join(gpus)}", flush=True)

    pending = list(CONFIGS)
    slots = {gpu: None for gpu in gpus}
    finished = []
    try:
        while pending or any(slot is not None for slot in slots.values()):
            for gpu in list(slots):
                slot = slots[gpu]
                if slot is not None and slot[0].poll() is not None:
                    process, handle, _log, _config = slot
                    handle.close()
                    finished.append((slot[3], process.returncode))
                    print(
                        f"done gpu={gpu} code={process.returncode} {slot[3]}",
                        flush=True,
                    )
                    slots[gpu] = None
                if slots[gpu] is None and pending:
                    config_path = pending.pop(0)
                    process, handle, log_path = start_job(python, gpu, config_path)
                    slots[gpu] = (process, handle, log_path, config_path)
            if any(slot is not None for slot in slots.values()):
                time.sleep(5)
    except BaseException:
        for slot in slots.values():
            if slot is not None and slot[0].poll() is None:
                slot[0].terminate()
        raise
    summary = summarize(finished)
    failed = [row for row in summary["runs"] if row["return_code"] != 0]
    if failed:
        raise SystemExit(f"{len(failed)} job(s) failed")


if __name__ == "__main__":
    main()
