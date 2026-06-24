#!/usr/bin/env python3
"""Prepare a tiny public-image dataset for Method 2.

This is intentionally small. It downloads a few public JPEG images, resizes
them to a fixed square size, and writes PPM files that the C++ Method 2 runner
can read without linking OpenCV or another image library.
"""

from __future__ import annotations

import argparse
import csv
import urllib.request
from pathlib import Path

import cv2
import numpy as np


DEFAULT_SOURCES = [
    ("bus", "https://ultralytics.com/images/bus.jpg"),
    ("zidane", "https://ultralytics.com/images/zidane.jpg"),
    ("dog", "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"),
    ("fruits", "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/fruits.jpg"),
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return repo_root() / path


def download_image(url: str) -> np.ndarray:
    with urllib.request.urlopen(url, timeout=30) as response:
        data = response.read()

    encoded = np.frombuffer(data, dtype=np.uint8)
    image_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"cannot decode image from {url}")

    return image_bgr


def write_ppm(path: Path, image_rgb: np.ndarray) -> None:
    height, width, channels = image_rgb.shape
    if channels != 3:
        raise RuntimeError("PPM writer expects RGB image")

    with path.open("wb") as f:
        f.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
        f.write(image_rgb.astype(np.uint8).tobytes())


def parse_sources(items: list[str]) -> list[tuple[str, str]]:
    if not items:
        return DEFAULT_SOURCES

    sources = []
    for item in items:
        if "=" not in item:
            raise SystemExit("--source must be formatted as name=url")
        name, url = item.split("=", 1)
        sources.append((name.strip(), url.strip()))
    return sources


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/vgg11-tiny-images")
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--source", action="append", default=[], help="Optional name=url image source.")
    args = parser.parse_args()

    if args.size <= 0:
        raise SystemExit("--size must be positive")
    if args.limit <= 0:
        raise SystemExit("--limit must be positive")

    output_dir = resolve_path(args.output_dir)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    list_lines = []
    sources = parse_sources(args.source)[: args.limit]

    for index, (name, url) in enumerate(sources):
        image_bgr = download_image(url)
        resized_bgr = cv2.resize(image_bgr, (args.size, args.size), interpolation=cv2.INTER_AREA)
        resized_rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)

        filename = f"{index:02d}_{name}_{args.size}x{args.size}.ppm"
        image_path = image_dir / filename
        write_ppm(image_path, resized_rgb)

        rel_image = image_path.relative_to(output_dir)
        list_lines.append(str(rel_image))
        rows.append(
            {
                "image_index": index,
                "image_path": str(rel_image),
                "name": name,
                "source_url": url,
                "height": args.size,
                "width": args.size,
            }
        )

    with (output_dir / "image_list.txt").open("w", encoding="utf-8") as f:
        f.write("\n".join(list_lines))
        f.write("\n")

    with (output_dir / "labels.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_index", "image_path", "name", "source_url", "height", "width"])
        writer.writeheader()
        writer.writerows(rows)

    with (output_dir / "manifest.txt").open("w", encoding="utf-8") as f:
        f.write("dataset=VGG11 tiny public-image smoke set\n")
        f.write(f"image_count={len(rows)}\n")
        f.write(f"shape={args.size}x{args.size} RGB\n")
        f.write("format=PPM P6\n")
        f.write("purpose=Small real-image input set for Method 2 VGG11 MPI convolution benchmark\n")
        f.write("sources=\n")
        for row in rows:
            f.write(f"- {row['name']}: {row['source_url']}\n")

    print("VGG11_TINY_IMAGES_DONE=YES")
    print(f"VGG11_TINY_IMAGES_DIR={output_dir}")
    print(f"VGG11_TINY_IMAGES_LIST={output_dir / 'image_list.txt'}")
    print(f"VGG11_TINY_IMAGES_COUNT={len(rows)}")


if __name__ == "__main__":
    main()
