from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Concatenate MOT17-style MP4 files into a fixed-length benchmark video."
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--frames", type=int, default=1200)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--source", action="append", required=True, help="Input MP4. Repeat this flag.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise SystemExit(f"Missing opencv-python: {exc}") from exc

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output), fourcc, args.fps, (args.width, args.height))
    if not writer.isOpened():
        raise SystemExit(f"Cannot open output video writer: {output}")

    written = 0
    used: list[tuple[str, int]] = []

    for source_text in args.source:
        source = Path(source_text)
        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            writer.release()
            raise SystemExit(f"Cannot open source video: {source}")

        local_count = 0
        while written < args.frames:
            ok, frame = cap.read()
            if not ok:
                break

            if frame.shape[1] != args.width or frame.shape[0] != args.height:
                frame = cv2.resize(frame, (args.width, args.height))

            writer.write(frame)
            written += 1
            local_count += 1

        cap.release()
        used.append((str(source), local_count))

        if written >= args.frames:
            break

    writer.release()

    if written != args.frames:
        raise SystemExit(f"Only wrote {written}/{args.frames} frames")

    print(f"BENCHMARK_VIDEO={output}")
    print(f"FRAMES={written}")
    for source, count in used:
        print(f"USED {source} {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
