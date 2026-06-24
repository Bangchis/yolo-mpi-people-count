#!/usr/bin/env python3
"""Download CIFAR-10 and export a tiny PPM subset for Method 2.

The Method 2 C++ runner intentionally avoids image libraries. PPM keeps the
input path simple: Python converts CIFAR-10 binary rows once, then C++ reads the
RGB pixels directly.
"""

from __future__ import annotations

import argparse
import csv
import tarfile
import urllib.request
from pathlib import Path


CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
LABEL_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return repo_root() / path


def download_if_needed(url: str, archive_path: Path) -> None:
    if archive_path.exists() and archive_path.stat().st_size > 0:
        return

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = archive_path.with_suffix(archive_path.suffix + ".tmp")

    print(f"CIFAR10_DOWNLOAD url={url}")
    print(f"CIFAR10_DOWNLOAD_TO path={archive_path}")

    with urllib.request.urlopen(url) as response, tmp_path.open("wb") as out:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)

    tmp_path.replace(archive_path)


def ppm_bytes_from_cifar_row(row: bytes) -> tuple[int, bytes]:
    if len(row) != 3073:
        raise ValueError(f"bad CIFAR-10 row length: {len(row)}")

    label = row[0]
    pixels = row[1:]
    red = pixels[0:1024]
    green = pixels[1024:2048]
    blue = pixels[2048:3072]

    rgb = bytearray()
    for i in range(1024):
        rgb.extend((red[i], green[i], blue[i]))

    return label, bytes(rgb)


def export_test_batch(archive_path: Path, output_dir: Path, count: int) -> None:
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    labels_path = output_dir / "labels.csv"
    list_path = output_dir / "image_list.txt"
    manifest_path = output_dir / "manifest.txt"

    with tarfile.open(archive_path, "r:gz") as tar:
        member = tar.getmember("cifar-10-batches-bin/test_batch.bin")
        batch = tar.extractfile(member)
        if batch is None:
            raise RuntimeError("cannot read CIFAR-10 test_batch.bin")

        rows = []
        list_lines = []
        for index in range(count):
            row = batch.read(3073)
            if not row:
                break

            label, rgb = ppm_bytes_from_cifar_row(row)
            filename = f"cifar10_test_{index:05d}_{LABEL_NAMES[label]}.ppm"
            image_path = image_dir / filename

            with image_path.open("wb") as out:
                out.write(b"P6\n32 32\n255\n")
                out.write(rgb)

            rel_image = image_path.relative_to(output_dir)
            list_lines.append(str(rel_image))
            rows.append(
                {
                    "image_index": index,
                    "image_path": str(rel_image),
                    "label_id": label,
                    "label_name": LABEL_NAMES[label],
                }
            )

    with list_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(list_lines))
        f.write("\n")

    with labels_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_index", "image_path", "label_id", "label_name"])
        writer.writeheader()
        writer.writerows(rows)

    with manifest_path.open("w", encoding="utf-8") as f:
        f.write("dataset=CIFAR-10 mini test subset\n")
        f.write(f"source_url={CIFAR10_URL}\n")
        f.write("source_split=test_batch.bin\n")
        f.write("source_shape=32x32 RGB\n")
        f.write(f"image_count={len(rows)}\n")
        f.write("format=PPM P6\n")
        f.write("purpose=Small real-image input set for Method 2 VGG11 MPI convolution benchmark\n")

    print("CIFAR10_MINI_DONE=YES")
    print(f"CIFAR10_MINI_DIR={output_dir}")
    print(f"CIFAR10_MINI_LIST={list_path}")
    print(f"CIFAR10_MINI_COUNT={len(rows)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=CIFAR10_URL)
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--output-dir", default="data/vgg11-cifar10-mini")
    parser.add_argument("--count", type=int, default=32)
    args = parser.parse_args()

    if args.count <= 0:
        raise SystemExit("--count must be positive")

    raw_dir = resolve_path(args.raw_dir)
    output_dir = resolve_path(args.output_dir)
    archive_path = raw_dir / "cifar-10-binary.tar.gz"

    download_if_needed(args.url, archive_path)
    export_test_batch(archive_path, output_dir, args.count)


if __name__ == "__main__":
    main()
