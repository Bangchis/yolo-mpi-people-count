from __future__ import annotations

import argparse
import base64
from pathlib import Path


def decode_jpeg(cv2, np, encoded: str):
    raw = base64.b64decode(encoded.encode("ascii"), validate=True)
    arr = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("cannot decode frame jpeg")
    return image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--display", type=int, choices=[0, 1], default=1)
    parser.add_argument("--window-name", default="YOLO MPI Live")
    parser.add_argument("--save-dir", default="")
    parser.add_argument("--max-width", type=int, default=1280)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import cv2  # type: ignore
    import numpy as np  # type: ignore

    save_dir = Path(args.save_dir) if args.save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    frame_id = -1
    width = 0
    height = 0
    count = 0
    encoded = ""
    boxes: list[tuple[float, float, float, float, float]] = []

    for raw in __import__("sys").stdin:
        line = raw.strip()
        if not line:
            continue
        if line == "QUIT":
            break
        parts = line.split(maxsplit=5)
        tag = parts[0]
        if tag == "FRAME":
            if len(parts) != 6:
                continue
            frame_id = int(parts[1])
            width = int(parts[2])
            height = int(parts[3])
            count = int(parts[4])
            encoded = parts[5]
            boxes = []
            continue
        if tag == "BOX":
            if len(parts) < 6:
                continue
            boxes.append(tuple(float(x) for x in parts[1:6]))
            continue
        if tag != "END_FRAME":
            continue

        frame = decode_jpeg(cv2, np, encoded)
        for x1, y1, x2, y2, conf in boxes:
            p1 = (max(0, int(round(x1))), max(0, int(round(y1))))
            p2 = (min(width - 1, int(round(x2))), min(height - 1, int(round(y2))))
            cv2.rectangle(frame, p1, p2, (40, 220, 90), 2)
            cv2.putText(
                frame,
                f"person {conf:.2f}",
                (p1[0], max(18, p1[1] - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (40, 220, 90),
                2,
                cv2.LINE_AA,
            )
        cv2.putText(
            frame,
            f"frame {frame_id}  people {count}",
            (16, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"frame {frame_id}  people {count}",
            (16, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (30, 30, 30),
            1,
            cv2.LINE_AA,
        )

        if save_dir:
            cv2.imwrite(str(save_dir / f"frame_{frame_id:06d}.jpg"), frame)
        if args.display:
            if args.max_width > 0 and frame.shape[1] > args.max_width:
                scale = args.max_width / float(frame.shape[1])
                frame = cv2.resize(frame, (args.max_width, int(frame.shape[0] * scale)))
            cv2.imshow(args.window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    if args.display:
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
