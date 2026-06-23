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
    parser = argparse.ArgumentParser(description="Plot static-vs-dynamic scheduler comparison.")
    parser.add_argument("--input", required=True, help="scheduler_comparison.csv")
    parser.add_argument("--output", required=True, help="Output PNG")
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError as exc:
        raise SystemExit("Missing matplotlib. Install with: pip install -e '.[helpers]'") from exc

    rows = read_rows(Path(args.input))
    labels = [row["schedule"] for row in rows]
    x = list(range(len(rows)))
    width = 0.36

    with_comm = [float(row["total_ms_with_comm"]) / 1000.0 for row in rows]
    without_comm = [float(row["total_ms_without_comm"]) / 1000.0 for row in rows]
    comm_total = [float(row["comm_s_total"]) for row in rows]
    idle_total = [float(row["idle_s_total"]) for row in rows]
    imbalance = [float(row["load_imbalance"]) for row in rows]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))

    axes[0].bar([i - width / 2 for i in x], with_comm, width=width, label="with comm")
    axes[0].bar([i + width / 2 for i in x], without_comm, width=width, label="without comm")
    axes[0].set_title("Runtime")
    axes[0].set_ylabel("Seconds")
    axes[0].set_xticks(x, labels)
    axes[0].grid(True, axis="y", alpha=0.25)
    axes[0].legend(loc="best")

    axes[1].bar([i - width / 2 for i in x], comm_total, width=width, label="communication")
    axes[1].bar([i + width / 2 for i in x], idle_total, width=width, label="idle")
    axes[1].set_title("Overhead by rank metrics")
    axes[1].set_ylabel("Total seconds")
    axes[1].set_xticks(x, labels)
    axes[1].grid(True, axis="y", alpha=0.25)
    axes[1].legend(loc="best")

    axes[2].bar(x, imbalance, color="#54a24b")
    axes[2].set_title("Load imbalance")
    axes[2].set_ylabel("Max compute / avg compute")
    axes[2].set_xticks(x, labels)
    axes[2].grid(True, axis="y", alpha=0.25)

    fig.suptitle("Scheduler comparison: static vs dynamic", y=1.03)
    fig.tight_layout()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170, bbox_inches="tight")
    print(f"SCHEDULER_COMPARISON_PLOT={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
