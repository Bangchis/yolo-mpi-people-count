#!/usr/bin/env python3
"""Plot Method 2 input-size selection results."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    rows = read_rows(args.input)
    for row in rows:
        row["height_int"] = int(row["height"])
        row["runtime_s"] = float(row["distributed_ms"]) / 1000.0
        row["halo_s"] = float(row["halo_ms"]) / 1000.0
        row["compute_s"] = float(row["compute_ms"]) / 1000.0

    modes = sorted({row["halo_mode"] for row in rows})
    colors = {
        "blocking": "#2f6fdd",
        "nonblocking": "#248f5a",
    }

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax_runtime, ax_ratio = axes

    for mode in modes:
        mode_rows = sorted([row for row in rows if row["halo_mode"] == mode], key=lambda row: row["height_int"])
        sizes = [row["height_int"] for row in mode_rows]
        runtimes = [row["runtime_s"] for row in mode_rows]
        halo_ratios = [
            row["halo_s"] / row["runtime_s"] if row["runtime_s"] > 0 else 0.0
            for row in mode_rows
        ]
        color = colors.get(mode)

        ax_runtime.plot(sizes, runtimes, marker="o", linewidth=2, label=mode, color=color)
        ax_ratio.plot(sizes, halo_ratios, marker="s", linewidth=2, label=mode, color=color)

    ax_runtime.set_title("Runtime vs Input Size")
    ax_runtime.set_xlabel("input height/width")
    ax_runtime.set_ylabel("seconds")
    ax_runtime.grid(True, alpha=0.3)
    ax_runtime.legend(fontsize=8)

    ax_ratio.set_title("Halo-Time Fraction")
    ax_ratio.set_xlabel("input height/width")
    ax_ratio.set_ylabel("halo_ms / distributed_ms")
    ax_ratio.grid(True, alpha=0.3)
    ax_ratio.legend(fontsize=8)

    fig.suptitle("Method 2 Input Size Selection", fontsize=13)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)


if __name__ == "__main__":
    main()
