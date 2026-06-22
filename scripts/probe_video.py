from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--format", choices=["env", "json"], default="env")
    args = parser.parse_args()

    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise SystemExit("Missing opencv-python. Install with: pip install -e '.[helpers]'") from exc

    source = Path(args.source)
    if not source.exists():
        raise SystemExit(f"Video source not found: {source}")
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {source}")
    info = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
    }
    cap.release()
    if args.format == "json":
        print(json.dumps(info, sort_keys=True))
    else:
        print(f"YOLO_FRAME_WIDTH={info['width']}")
        print(f"YOLO_FRAME_HEIGHT={info['height']}")
        print(f"YOLO_VIDEO_FPS={info['fps']}")
        print(f"YOLO_VIDEO_TOTAL_FRAMES={info['frames']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

