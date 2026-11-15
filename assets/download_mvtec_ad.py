"""从 ModelScope 下载 MVTec AD 数据集。"""

import argparse
from pathlib import Path

from modelscope import dataset_snapshot_download


def download_mvtec_ad(dataset_id, output_dir):
    # dataset_id 使用 ModelScope 上实际存在的 namespace/name。
    return dataset_snapshot_download(
        dataset_id=dataset_id,
        local_dir=str(Path(output_dir)),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--output-dir", default="assets/datasets/mvtec_ad")
    args = parser.parse_args()
    download_mvtec_ad(args.dataset_id, args.output_dir)
