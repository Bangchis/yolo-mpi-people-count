from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download


DEFAULT_ASSETS = [
    "models/yolo11n.pt",
    "data/smoke_people.mp4",
    "data/bus.jpg",
    "data/mot17-mini/MOT17-02-SDP-300_960x540.mp4",
    "data/mot17-mini/MOT17-02-SDP-300_counts.csv",
    "data/mot17-mini/README.md",
    "data/mot17-fullseq/MOT17-02-SDP-600_960x540.mp4",
    "data/mot17-fullseq/MOT17-02-SDP-600_counts.csv",
    "data/mot17-fullseq/README.md",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download runtime assets from a Hugging Face repo.")
    parser.add_argument("--repo-id", required=True, help="Hugging Face repo id, for example USER/yolo-mpi-people-count-assets.")
    parser.add_argument("--repo-type", default="dataset", choices=["dataset", "model"])
    parser.add_argument("--revision", default="main")
    parser.add_argument("--asset", action="append", default=[], help="Remote asset path to download.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    assets = args.asset or DEFAULT_ASSETS
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
