from __future__ import annotations

import argparse
import base64
import sys
import time
from pathlib import Path


def fail(message: str) -> None:
    print(f"ERROR {message}", flush=True)
    raise SystemExit(1)


def parse_tile_grid(value: str) -> tuple[int, int]:
    sep = "x" if "x" in value else "X"
    if sep not in value:
        fail("tile grid must be COLSxROWS")
    cols, rows = (int(part) for part in value.split(sep, 1))
    if cols <= 0 or rows <= 0:
        fail("tile grid dimensions must be positive")
    return cols, rows


def make_tiles(width: int, height: int, cols: int, rows: int, overlap: int):
    tile_id = 0
    for row in range(rows):
        base_y1 = round(row * height / rows)
        base_y2 = round((row + 1) * height / rows)
        for col in range(cols):
            base_x1 = round(col * width / cols)
            base_x2 = round((col + 1) * width / cols)
            yield (
                tile_id,
                max(0, base_x1 - overlap),
                max(0, base_y1 - overlap),
                min(width, base_x2 + overlap),
                min(height, base_y2 + overlap),
            )
            tile_id += 1


def encode_jpeg(cv2, image, quality: int) -> str:
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--video-source", default="")
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--tile-grid", default="1x1")
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument("--jpeg-quality", type=int, default=80)
    parser.add_argument("--target-fps", type=float, default=0.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        fail(f"missing opencv-python: {exc}")

    if args.video_source:
        path = Path(args.video_source)
        if not path.exists():
            fail(f"video source not found: {path}")
        source = str(path)
        source_label = str(path)
    else:
        source = args.camera_index
        source_label = f"camera:{args.camera_index}"

    if args.video_source:
        cap = cv2.VideoCapture(source)
    else:
        # macOS camera access is most reliable when OpenCV uses AVFoundation
        # directly instead of probing generic video backends first.
        cap = cv2.VideoCapture(source, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        fail(f"cannot open {source_label}")

    if args.video_source and args.start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
    elif not args.video_source:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    cols, rows = parse_tile_grid(args.tile_grid)
    print(
        f"READY source={source_label} width={args.width} height={args.height} "
        f"tile_grid={args.tile_grid} overlap={args.overlap}",
        flush=True,
    )

    frame_id = args.start_frame
    emitted = 0
    next_frame_at = time.perf_counter()
    try:
        while args.frames == 0 or emitted < args.frames:
            if args.target_fps > 0:
                delay = next_frame_at - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                next_frame_at = max(next_frame_at + 1.0 / args.target_fps, time.perf_counter())

            t0 = time.perf_counter()
            ok, frame = cap.read()
            capture_ms = (time.perf_counter() - t0) * 1000.0
            if not ok:
                print(f"END_STREAM frame_id={frame_id} emitted={emitted}", flush=True)
                break

            if args.width > 0 and args.height > 0:
                frame = cv2.resize(frame, (args.width, args.height), interpolation=cv2.INTER_AREA)
            height, width = frame.shape[:2]
            full_jpeg = encode_jpeg(cv2, frame, args.jpeg_quality)
            print(f"FRAME {frame_id} {width} {height} {capture_ms:.4f} {full_jpeg}", flush=False)

            task_id_base = emitted * cols * rows
            tile_count = 0
            for tile_id, x1, y1, x2, y2 in make_tiles(width, height, cols, rows, args.overlap):
                tile = frame[y1:y2, x1:x2]
                tile_jpeg = encode_jpeg(cv2, tile, args.jpeg_quality)
                task_id = task_id_base + tile_id
                print(f"TILE {task_id} {frame_id} {tile_id} {x1} {y1} {x2} {y2} {tile_jpeg}", flush=False)
                tile_count += 1

            print(f"END_FRAME {frame_id} {tile_count}", flush=True)
            frame_id += 1
            emitted += 1
    except BrokenPipeError:
        return 0
    finally:
        cap.release()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
