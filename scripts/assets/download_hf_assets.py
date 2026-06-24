from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download


DEFAULT_ASSETS = [
    "models/yolo11n.pt",
    "data/smoke_people.mp4",
    "data/bus.jpg",
    "data/mot17-mini/MOT17-02-SDP-300_960x540.mp4",
    "data/mot17-mini/MOT17-02-SDP-300_counts.csv",
    "data/mot17-mini/README.md",
    "data/mot17-fullseq/MOT17-02-SDP-600_960x540.mp4",
    "data/mot17-fullseq/MOT17-02-SDP-600_counts.csv",
    "data/mot17-fullseq/MOT17-05-SDP-837_960x540.mp4",
    "data/mot17-fullseq/MOT17-05-SDP-837_counts.csv",
    "data/mot17-fullseq/MOT17-09-SDP-525_960x540.mp4",
    "data/mot17-fullseq/MOT17-09-SDP-525_counts.csv",
    "data/mot17-fullseq/MOT17-10-SDP-654_960x540.mp4",
    "data/mot17-fullseq/MOT17-10-SDP-654_counts.csv",
    "data/mot17-fullseq/README.md",
]

DEFAULT_FOLDERS = [
    "data/vgg11-tiny-images",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download runtime assets from a Hugging Face repo.")
    parser.add_argument("--repo-id", required=True, help="Hugging Face repo id, for example USER/yolo-mpi-people-count-assets.")
    parser.add_argument("--repo-type", default="dataset", choices=["dataset", "model"])
    parser.add_argument("--revision", default="main")
    parser.add_argument("--asset", action="append", default=[], help="Remote asset path to download.")
    parser.add_argument("--folder", action="append", default=[], help="Remote folder path to download.")
    parser.add_argument("--no-defaults", action="store_true", help="Download only explicit --asset/--folder entries.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    assets = args.asset if args.no_defaults else (args.asset or DEFAULT_ASSETS)
    folders = args.folder if args.no_defaults else (args.folder or DEFAULT_FOLDERS)

    for remote_path in assets:
        local_path = Path(remote_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        downloaded = hf_hub_download(
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            revision=args.revision,
            filename=remote_path,
            local_dir=".",
        )
        print(f"DOWNLOADED {remote_path} -> {downloaded}")

    for remote_folder in folders:
        snapshot_dir = snapshot_download(
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            revision=args.revision,
            allow_patterns=[remote_folder.rstrip("/") + "/**"],
            local_dir=".",
        )
        print(f"DOWNLOADED_FOLDER {remote_folder} -> {snapshot_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
