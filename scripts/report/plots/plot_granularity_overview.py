from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"No rows in {path}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot a compact overview for tile granularity experiments.")
    parser.add_argument("--input", required=True, help="granularity_overview.csv")
    parser.add_argument("--output", required=True, help="Output PNG")
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError as exc:
        raise SystemExit("Missing matplotlib. Install with: pip install -e '.[helpers]'") from exc

    rows = read_rows(Path(args.input))
    labels = [row["label"].replace("grid_", "") for row in rows]
    tasks = [int(float(row["total_tasks"])) for row in rows]
    compute_max = [float(row["compute_s_max"]) for row in rows]
    compute_avg = [float(row["compute_s_avg"]) for row in rows]
    comm_total = [float(row["comm_s_total"]) for row in rows]
    idle_gap = [float(row["idle_gap_ratio"]) for row in rows]

    x = list(range(len(rows)))
    width = 0.36

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))

    axes[0].bar([i - width / 2 for i in x], compute_max, width=width, label="max compute")
    axes[0].bar([i + width / 2 for i in x], compute_avg, width=width, label="avg compute")
    axes[0].set_title("Compute by tile grid")
    axes[0].set_ylabel("Seconds")
    axes[0].set_xticks(x, labels)
    axes[0].grid(True, axis="y", alpha=0.25)
    axes[0].legend(loc="best")

    axes[1].bar(x, comm_total, color="#4c78a8")
    axes[1].set_title("Communication overhead")
    axes[1].set_ylabel("Total seconds")
    axes[1].set_xticks(x, labels)
    axes[1].grid(True, axis="y", alpha=0.25)

    axes[2].plot(x, idle_gap, marker="o", label="idle gap ratio")
    axes[2].axhline(0.25, linestyle="--", color="#d95f02", label="25% threshold")
    for i, task_count in enumerate(tasks):
        axes[2].annotate(str(task_count), (i, idle_gap[i]), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
    axes[2].set_title("Load-balance criterion")
    axes[2].set_ylabel("Idle gap / max compute")
    axes[2].set_xticks(x, labels)
    axes[2].grid(True, axis="y", alpha=0.25)
    axes[2].legend(loc="best")

    fig.suptitle("Granularity sweep: few tile variants, clear trade-offs", y=1.03)
    fig.tight_layout()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170, bbox_inches="tight")
    print(f"GRANULARITY_OVERVIEW_PLOT={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
