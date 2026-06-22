from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolo11n.pt", help="Ultralytics model name to download.")
    parser.add_argument("--output", default="models/yolo11n.pt", help="Local output model path.")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError as exc:
        raise SystemExit("Missing ultralytics. Install with: pip install -e '.[yolo]'") from exc

    output = Path(args.output)
    if output.exists():
        print(f"MODEL_READY={output}")
        return 0

    output.parent.mkdir(parents=True, exist_ok=True)
    cwd = Path.cwd()
    model_name = Path(args.model).name
    try:
        # Ultralytics downloads known pretrained names into the current working directory
        # on first use. Run this inside models/ so the artifact is easy to sync.
        import os

        os.chdir(output.parent)
        YOLO(model_name)
    finally:
        os.chdir(cwd)

    downloaded = output.parent / model_name
    if not downloaded.exists():
        raise SystemExit(f"Model download did not produce {downloaded}")
    if downloaded.resolve() != output.resolve():
        shutil.move(str(downloaded), str(output))
    print(f"MODEL_READY={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

