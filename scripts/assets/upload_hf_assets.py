from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path

from huggingface_hub import HfApi


DEFAULT_ASSETS = [
    ("models/yolo11n.pt", "models/yolo11n.pt"),
    ("data/smoke_people.mp4", "data/smoke_people.mp4"),
    ("data/bus.jpg", "data/bus.jpg"),
    ("data/mot17-mini/MOT17-02-SDP-300_960x540.mp4", "data/mot17-mini/MOT17-02-SDP-300_960x540.mp4"),
    ("data/mot17-mini/MOT17-02-SDP-300_counts.csv", "data/mot17-mini/MOT17-02-SDP-300_counts.csv"),
    ("data/mot17-mini/README.md", "data/mot17-mini/README.md"),
    ("data/mot17-fullseq/MOT17-02-SDP-600_960x540.mp4", "data/mot17-fullseq/MOT17-02-SDP-600_960x540.mp4"),
    ("data/mot17-fullseq/MOT17-02-SDP-600_counts.csv", "data/mot17-fullseq/MOT17-02-SDP-600_counts.csv"),
    ("data/mot17-fullseq/MOT17-05-SDP-837_960x540.mp4", "data/mot17-fullseq/MOT17-05-SDP-837_960x540.mp4"),
    ("data/mot17-fullseq/MOT17-05-SDP-837_counts.csv", "data/mot17-fullseq/MOT17-05-SDP-837_counts.csv"),
    ("data/mot17-fullseq/MOT17-09-SDP-525_960x540.mp4", "data/mot17-fullseq/MOT17-09-SDP-525_960x540.mp4"),
    ("data/mot17-fullseq/MOT17-09-SDP-525_counts.csv", "data/mot17-fullseq/MOT17-09-SDP-525_counts.csv"),
    ("data/mot17-fullseq/MOT17-10-SDP-654_960x540.mp4", "data/mot17-fullseq/MOT17-10-SDP-654_960x540.mp4"),
    ("data/mot17-fullseq/MOT17-10-SDP-654_counts.csv", "data/mot17-fullseq/MOT17-10-SDP-654_counts.csv"),
    ("data/mot17-fullseq/README.md", "data/mot17-fullseq/README.md"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload local runtime assets to a Hugging Face dataset repo.")
    parser.add_argument("--repo-id", required=True, help="Hugging Face repo id, for example USER/yolo-mpi-people-count-assets.")
    parser.add_argument("--repo-type", default="dataset", choices=["dataset", "model"], help="Repo type to create/upload to.")
    parser.add_argument("--private", action="store_true", help="Create the Hugging Face repo as private.")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--asset", action="append", default=[], help="Extra LOCAL_PATH:REMOTE_PATH asset mapping.")
    return parser.parse_args()


def parse_asset_mapping(value: str) -> tuple[str, str]:
    if ":" not in value:
        return value, value
    local, remote = value.split(":", 1)
    return local, remote


def main() -> int:
    args = parse_args()
    api = HfApi()
    api.create_repo(repo_id=args.repo_id, repo_type=args.repo_type, private=args.private, exist_ok=True)

    assets = list(DEFAULT_ASSETS)
    assets.extend(parse_asset_mapping(item) for item in args.asset)

    for local, remote in assets:
        path = Path(local)
        if not path.exists():
            print(f"SKIP missing {local}")
            continue
        api.upload_file(
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            revision=args.revision,
            path_or_fileobj=str(path),
            path_in_repo=remote,
            commit_message=f"Upload {remote}",
        )
        print(f"UPLOADED {local} -> hf://{args.repo_id}/{remote}")

    contents = "\n".join(f"- `{remote}`" for _, remote in assets)
    readme = (
        "---\n"
        "license: mit\n"
        "task_categories:\n"
        "- object-detection\n"
        "pretty_name: YOLO MPI People Count Assets\n"
        "---\n\n"
        "# YOLO MPI People Count Assets\n\n"
        "Runtime assets for `Bangchis/yolo-mpi-people-count`.\n\n"
        "Contents:\n"
        f"{contents}\n\n"
        "No cluster secrets, SSH keys, local `configs/cluster_macos.env`, build output, or result logs are included.\n"
    )
    api.upload_file(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        revision=args.revision,
        path_or_fileobj=BytesIO(readme.encode("utf-8")),
        path_in_repo="README.md",
        commit_message="Add asset card",
    )
    print(f"HF_ASSETS_REPO={args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
