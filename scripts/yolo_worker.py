from __future__ import annotations

import argparse
import base64
import socket
import sys
import time
from pathlib import Path


def fail(message: str, code: int = 1) -> None:
    print(f"ERROR {message}", flush=True)
    raise SystemExit(code)


def load_dependencies():
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        fail(f"missing opencv-python: {exc}")
    try:
        import numpy as np  # type: ignore
    except ImportError as exc:
        fail(f"missing numpy: {exc}")
    try:
        import torch  # type: ignore
    except ImportError as exc:
        fail(f"missing torch: {exc}")
    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError as exc:
        fail(f"missing ultralytics: {exc}")
    return cv2, np, torch, YOLO


def choose_device(torch, requested: str, cpu_fallback: bool) -> str:
    if requested != "mps":
        return requested
    ok = bool(torch.backends.mps.is_built() and torch.backends.mps.is_available())
    if ok:
        return "mps"
    if cpu_fallback:
        print("WARN mps unavailable; falling back to cpu", file=sys.stderr, flush=True)
        return "cpu"
    fail("mps unavailable and cpu fallback disabled")
    return "cpu"


def read_frame(cv2, cap, source: str, frame_id: int, cache: dict[str, object]):
    if cap is None:
        raise RuntimeError("TASK requires a readable video source; use IMAGE for live tiles")
    if cache.get("frame_id") == frame_id:
        return cache["frame"]
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"cannot read frame {frame_id} from {source}")
    cache["frame_id"] = frame_id
    cache["frame"] = frame
    return frame


def decode_image(cv2, np, encoded: str):
    raw = base64.b64decode(encoded.encode("ascii"), validate=True)
    arr = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("cannot decode IMAGE payload")
    return image


def predict_tile(model, tile, device: str, args: argparse.Namespace):
    start = time.perf_counter()
    results = model.predict(
        tile,
        device=device,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        classes=[args.class_id],
        verbose=False,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    detections: list[tuple[float, float, float, float, float]] = []
    if results:
        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().tolist()
            confs = boxes.conf.cpu().tolist()
            classes = boxes.cls.cpu().tolist()
            for coords, confidence, cls in zip(xyxy, confs, classes):
                if int(cls) != args.class_id:
                    continue
                bx1, by1, bx2, by2 = coords
                detections.append((float(bx1), float(by1), float(bx2), float(by2), float(confidence)))
    return detections, elapsed_ms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="")
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--class-id", type=int, default=0)
    parser.add_argument("--cpu-fallback", type=int, choices=[0, 1], default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cv2, np, torch, YOLO = load_dependencies()
    source = Path(args.source) if args.source else None
    model_path = Path(args.model)
    model_arg = str(model_path) if model_path.exists() else args.model

    device = choose_device(torch, args.device, bool(args.cpu_fallback))
    cap = None
    if source is not None and source.exists():
        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            fail(f"cannot open video: {source}")
    model = YOLO(model_arg)
    cache: dict[str, object] = {}
    host = socket.gethostname()
    print(f"READY host={host} device={device} model={model_arg}", flush=True)

    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        if line == "QUIT":
            print("BYE", flush=True)
            break
        parts = line.split(maxsplit=8)
        if parts[0] not in {"TASK", "IMAGE"}:
            print(f"ERROR malformed task line: {line}", flush=True)
            continue

        if parts[0] == "TASK" and len(parts) != 8:
            print(f"ERROR malformed TASK line: {line}", flush=True)
            continue
        if parts[0] == "IMAGE" and len(parts) != 9:
            print("ERROR malformed IMAGE line", flush=True)
            continue

        task_id, frame_id, tile_id, x1, y1, x2, y2 = map(int, parts[1:8])
        try:
            if parts[0] == "TASK":
                frame = read_frame(cv2, cap, str(source), frame_id, cache)
                tile = frame[y1:y2, x1:x2]
            else:
                tile = decode_image(cv2, np, parts[8])
            detections, elapsed_ms = predict_tile(model, tile, device, args)
            print(f"BEGIN {task_id}", flush=False)
            for bx1, by1, bx2, by2, confidence in detections:
                print(f"DET {bx1:.4f} {by1:.4f} {bx2:.4f} {by2:.4f} {confidence:.6f}", flush=False)
            print(f"END {task_id} {elapsed_ms:.4f}", flush=True)
        except Exception as exc:  # noqa: BLE001 - worker protocol should report task failures.
            print(f"ERROR task_id={task_id} {type(exc).__name__}: {exc}", flush=True)

    if cap is not None:
        cap.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
