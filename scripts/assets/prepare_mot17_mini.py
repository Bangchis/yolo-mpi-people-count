from __future__ import annotations

import argparse
import csv
import shutil
import zipfile
from pathlib import Path


def require_cv2():
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise SystemExit("Missing opencv-python. Install with: pip install -e '.[helpers]'") from exc
    return cv2


def parse_gt(text: str, start_frame: int, frames: int, visibility_threshold: float) -> dict[int, int]:
    counts = {frame_id: 0 for frame_id in range(frames)}
    end_frame = start_frame + frames - 1
    for raw in text.splitlines():
        if not raw.strip():
            continue
        parts = raw.split(",")
        if len(parts) < 9:
            continue
        mot_frame = int(float(parts[0]))
        if mot_frame < start_frame or mot_frame > end_frame:
            continue
        conf = float(parts[6])
        class_id = int(float(parts[7]))
        visibility = float(parts[8])
        if conf <= 0 or class_id != 1 or visibility < visibility_threshold:
            continue
        counts[mot_frame - start_frame] += 1
    return counts


def find_sequence_dir(root: Path, sequence: str) -> Path:
    candidates = [
        root / sequence,
        root / "train" / sequence,
        root / "MOT17" / "train" / sequence,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise SystemExit(f"Cannot find sequence {sequence} under {root}")


def prepare_from_root(args: argparse.Namespace, cv2) -> tuple[Path, dict[int, int]]:
    sequence_dir = find_sequence_dir(Path(args.mot17_root), args.sequence)
    gt_path = sequence_dir / "gt" / "gt.txt"
    image_dir = sequence_dir / "img1"
    if not gt_path.exists():
        raise SystemExit(f"Missing ground truth: {gt_path}")
    if not image_dir.exists():
        raise SystemExit(f"Missing image directory: {image_dir}")

    video_path = write_video_from_images(
        cv2=cv2,
        image_reader=lambda mot_frame: cv2.imread(str(image_dir / f"{mot_frame:06d}.jpg")),
        args=args,
    )
    counts = parse_gt(gt_path.read_text(encoding="utf-8"), args.start_frame, args.frames, args.visibility_threshold)
    return video_path, counts


def zip_name_lookup(zf: zipfile.ZipFile) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for name in zf.namelist():
        lookup[name.lower()] = name
    return lookup


def find_zip_member(lookup: dict[str, str], suffix: str) -> str:
    suffix = suffix.lower().replace("\\", "/")
    for lower_name, original in lookup.items():
        if lower_name.endswith(suffix):
            return original
    raise SystemExit(f"Cannot find {suffix} in MOT17 zip")


def prepare_from_zip(args: argparse.Namespace, cv2) -> tuple[Path, dict[int, int]]:
    import numpy as np  # type: ignore

    zip_path = Path(args.mot17_zip)
    if not zip_path.exists():
        raise SystemExit(f"Missing MOT17 zip: {zip_path}")
    with zipfile.ZipFile(zip_path) as zf:
        lookup = zip_name_lookup(zf)
        gt_member = find_zip_member(lookup, f"{args.sequence}/gt/gt.txt")
        gt_text = zf.read(gt_member).decode("utf-8")

        def read_image(mot_frame: int):
            image_member = find_zip_member(lookup, f"{args.sequence}/img1/{mot_frame:06d}.jpg")
            raw = zf.read(image_member)
            arr = np.frombuffer(raw, dtype=np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)

        video_path = write_video_from_images(cv2=cv2, image_reader=read_image, args=args)
    counts = parse_gt(gt_text, args.start_frame, args.frames, args.visibility_threshold)
    return video_path, counts


def write_video_from_images(cv2, image_reader, args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / f"{args.sequence}-{args.frames}_{args.resize_width}x{args.resize_height}.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (args.resize_width, args.resize_height),
    )
    if not writer.isOpened():
        raise SystemExit(f"Cannot create video: {video_path}")
    try:
        for offset in range(args.frames):
            mot_frame = args.start_frame + offset
            frame = image_reader(mot_frame)
            if frame is None:
                raise SystemExit(f"Cannot read MOT17 frame {mot_frame:06d}.jpg")
            resized = cv2.resize(frame, (args.resize_width, args.resize_height), interpolation=cv2.INTER_AREA)
            writer.write(resized)
    finally:
        writer.release()
    return video_path


def write_counts(path: Path, counts: dict[int, int], start_frame: int) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_id", "mot_frame_id", "person_count"])
        for frame_id in sorted(counts):
            writer.writerow([frame_id, start_frame + frame_id, counts[frame_id]])


def write_readme(path: Path, args: argparse.Namespace, video_path: Path, counts_path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "# MOT17 Mini Dataset",
                "",
                f"- Source sequence: `{args.sequence}`",
                f"- Frames: `{args.start_frame}` to `{args.start_frame + args.frames - 1}`",
                f"- Output video: `{video_path.name}`",
                f"- Ground-truth counts: `{counts_path.name}`",
                f"- Resize: `{args.resize_width}x{args.resize_height}`",
                f"- Visibility threshold: `{args.visibility_threshold}`",
                "",
                "This mini asset is used for the YOLO MPI people-counting report benchmarks.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a small MOT17 video and count ground truth for report benchmarks.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--mot17-root", help="Path to MOT17 root or train directory")
    source.add_argument("--mot17-zip", help="Path to MOT17.zip")
    parser.add_argument("--sequence", default="MOT17-02-SDP")
    parser.add_argument("--start-frame", type=int, default=1)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--resize-width", type=int, default=960)
    parser.add_argument("--resize-height", type=int, default=540)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--visibility-threshold", type=float, default=0.25)
    parser.add_argument("--output-dir", default="data/mot17-mini")
    parser.add_argument(
        "--delete-source-after",
        action="store_true",
        help="Delete the provided zip or root after creating the mini dataset. Use only after verifying upload.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cv2 = require_cv2()
    if args.mot17_zip:
        video_path, counts = prepare_from_zip(args, cv2)
        source_path = Path(args.mot17_zip)
    else:
        video_path, counts = prepare_from_root(args, cv2)
        source_path = Path(args.mot17_root)

    output_dir = Path(args.output_dir)
    counts_path = output_dir / f"{args.sequence}-{args.frames}_counts.csv"
    write_counts(counts_path, counts, args.start_frame)
    write_readme(output_dir / "README.md", args, video_path, counts_path)

    if args.delete_source_after:
        if source_path.is_dir():
            shutil.rmtree(source_path)
        else:
            source_path.unlink()

    print(f"MOT17_MINI_VIDEO={video_path}")
    print(f"MOT17_MINI_COUNTS={counts_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
