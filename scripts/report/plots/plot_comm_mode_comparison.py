#!/usr/bin/env python3
"""Plot MPI communication-mode benchmark results."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot MPI communication-mode comparison.")
    parser.add_argument("--input", required=True, help="comm_mode_overview.csv")
    parser.add_argument("--output", required=True, help="output PNG")
    args = parser.parse_args()

    rows = read_rows(Path(args.input))
    preferred = ["blocking", "nonblocking", "streaming"]
    available = {row["comm_mode"] for row in rows}
    modes = [mode for mode in preferred if mode in available]
    labels = ["Find N runtime", "Granularity runtime", "Speedup P=12 runtime"]
    keys = ["find_n_600_s", "granularity_5x4_s", "speedup_p12_s"]

    by_mode = {row["comm_mode"]: row for row in rows}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    x = range(len(labels))
    width = 0.75 / max(1, len(modes))
    start = -0.5 * width * (len(modes) - 1)

    offsets = [start + i * width for i in range(len(modes))]
    for offset, mode in zip(offsets, modes):
        values = [float(by_mode.get(mode, {}).get(key, 0) or 0) for key in keys]
        axes[0].bar([i + offset for i in x], values, width=width, label=mode)

    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(labels, rotation=20, ha="right")
    axes[0].set_ylabel("Runtime with communication (s)")
    axes[0].set_title("Runtime Comparison")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)

    speedups = [float(by_mode.get(mode, {}).get("speedup_p12", 0) or 0) for mode in modes]
    axes[1].bar(modes, speedups, color=["#4C78A8", "#F58518", "#54A24B"][: len(modes)])
    axes[1].set_ylabel("Speedup at P=12")
    axes[1].set_title("Speedup Comparison")
    axes[1].grid(axis="y", alpha=0.25)

    fig.tight_layout()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
