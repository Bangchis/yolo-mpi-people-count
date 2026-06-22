from __future__ import annotations

import argparse
import csv
from pathlib import Path


def require_cv2():
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise SystemExit("Missing opencv-python. Install with: pip install -e '.[helpers]'") from exc
    return cv2


def read_boxes(path: Path) -> dict[int, list[dict[str, float]]]:
    grouped: dict[int, list[dict[str, float]]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            frame_id = int(row["frame_id"])
            grouped.setdefault(frame_id, []).append(
                {
                    "x1": float(row["x1"]),
                    "y1": float(row["y1"]),
                    "x2": float(row["x2"]),
                    "y2": float(row["y2"]),
                    "conf": float(row["conf"]),
                }
            )
    return grouped


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--bboxes", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--start-frame", type=int, default=0)
    args = parser.parse_args()

    source = Path(args.source)
    if not source.exists():
        raise SystemExit(f"Video source not found: {source}")
    bboxes = Path(args.bboxes)
    if not bboxes.exists():
        raise SystemExit(f"bboxes.csv not found: {bboxes}")

    cv2 = require_cv2()
    boxes_by_frame = read_boxes(bboxes)
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {source}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise SystemExit(f"Cannot create output video: {output}")

    try:
        for frame_id in range(args.start_frame, args.start_frame + args.frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ok, frame = cap.read()
            if not ok:
                break
            boxes = boxes_by_frame.get(frame_id, [])
            for box in boxes:
                x1, y1, x2, y2 = map(int, [box["x1"], box["y1"], box["x2"], box["y2"]])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 210, 80), 2)
                cv2.putText(
                    frame,
                    f"person {box['conf']:.2f}",
                    (x1, max(20, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 210, 80),
                    2,
                )
            cv2.putText(
                frame,
                f"frame={frame_id} count={len(boxes)}",
                (16, 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (40, 230, 255),
                2,
            )
            writer.write(frame)
    finally:
        cap.release()
        writer.release()

    print(f"DEMO_VIDEO={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
