"""上传整理后的 can_detect 数据集，不下载任何模型或数据。"""

import argparse

from modelscope_hub import HubApi
from modelscope_hub.constants import Visibility


def upload(folder, token, repo_id):
    api = HubApi(token=token)
    if not api.repo_exists(repo_id, "dataset"):
        api.create_repo(
            repo_id,
            "dataset",
            visibility=Visibility.PUBLIC,
            description="can_detect clean five-fold dataset with pinyin can names and 3DGS reference gallery",
        )
    return api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=folder,
        commit_message="整理 can_detect 实验数据",
        max_workers=4,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--repo-id", default="Sondottt/can_detect")
    args = parser.parse_args()
    print(upload(args.folder, args.token, args.repo_id))
