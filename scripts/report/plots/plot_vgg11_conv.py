#!/usr/bin/env python3
"""Plot Method 2 VGG11 distributed convolution benchmark results."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_layer_breakdown(raw_dir: Path, halo_mode: str, p: int) -> dict[str, float]:
    rows = read_csv(raw_dir / halo_mode / f"p_{p}" / "layer_metrics.csv")
    totals = {
        "scatter_ms": 0.0,
        "halo_ms": 0.0,
        "compute_ms": 0.0,
        "gather_ms": 0.0,
    }

    for row in rows:
        for key in totals:
            totals[key] += float(row[key])

    return totals


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--speedup", required=True, type=Path)
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    rows = read_csv(args.speedup)
    for row in rows:
        row["p_int"] = int(row["p"])
        row["halo_mode"] = row.get("halo_mode", "blocking")

    modes = sorted({row["halo_mode"] for row in rows})
    all_p_values = sorted({row["p_int"] for row in rows})
    mode_rows = {
        mode: sorted([row for row in rows if row["halo_mode"] == mode], key=lambda row: row["p_int"])
        for mode in modes
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax_runtime, ax_speedup, ax_breakdown, ax_eff = axes.flatten()

    colors = {
        "blocking": "#2f6fdd",
        "nonblocking": "#248f5a",
    }

    for mode in modes:
        p_values = [row["p_int"] for row in mode_rows[mode]]
        runtimes = [float(row["distributed_ms"]) / 1000.0 for row in mode_rows[mode]]
        speedups = [float(row["speedup"]) for row in mode_rows[mode]]
        efficiencies = [float(row["efficiency"]) for row in mode_rows[mode]]
        inter_ratios = [float(row.get("halo_inter_machine_edge_ratio", 0.0)) for row in mode_rows[mode]]
        color = colors.get(mode, None)

        ax_runtime.plot(p_values, runtimes, marker="o", linewidth=2, label=mode, color=color)
        ax_speedup.plot(p_values, speedups, marker="o", linewidth=2, label=mode, color=color)
        ax_eff.plot(p_values, efficiencies, marker="o", linewidth=2, color=color, label=f"{mode} efficiency")
        ax_eff.plot(p_values,
                    inter_ratios,
                    marker="s",
                    linestyle="--",
                    linewidth=1.5,
                    color=color,
                    label=f"{mode} inter-machine ratio")

    ax_runtime.set_title("Runtime")
    ax_runtime.set_xlabel("MPI processes")
    ax_runtime.set_ylabel("seconds")
    ax_runtime.grid(True, alpha=0.3)
    ax_runtime.legend(fontsize=8)

    ax_speedup.plot(all_p_values, all_p_values, linestyle="--", label="Ideal", color="#888888")
    ax_speedup.set_title("Speedup")
    ax_speedup.set_xlabel("MPI processes")
    ax_speedup.set_ylabel("speedup")
    ax_speedup.grid(True, alpha=0.3)
    ax_speedup.legend()

    breakdown_labels = []
    breakdowns = []
    for mode in modes:
        if not mode_rows[mode]:
            continue
        largest_p = max(row["p_int"] for row in mode_rows[mode])
        breakdown_labels.append(f"{mode}\nP={largest_p}")
        breakdowns.append(read_layer_breakdown(args.raw_dir, mode, largest_p))

    bottom = [0.0 for _ in breakdowns]
    for key, label, color in [
        ("scatter_ms", "scatter", "#6c8ebf"),
        ("halo_ms", "halo exchange", "#d6b656"),
        ("compute_ms", "compute", "#82b366"),
        ("gather_ms", "gather", "#b85450"),
    ]:
        values = [b[key] / 1000.0 for b in breakdowns]
        ax_breakdown.bar(breakdown_labels, values, bottom=bottom, label=label, color=color)
        bottom = [a + b for a, b in zip(bottom, values)]

    ax_breakdown.set_title("Layer Time Breakdown")
    ax_breakdown.set_xlabel("MPI processes")
    ax_breakdown.set_ylabel("seconds")
    ax_breakdown.legend(fontsize=8)

    ax_eff.set_title("Efficiency and Topology Cost")
    ax_eff.set_xlabel("MPI processes")
    ax_eff.set_ylabel("ratio")
    ax_eff.set_ylim(bottom=0)
    ax_eff.grid(True, alpha=0.3)
    ax_eff.legend(fontsize=8)

    fig.suptitle("Method 2: VGG11 No-BN Distributed Convolution", fontsize=14)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)


if __name__ == "__main__":
    main()
