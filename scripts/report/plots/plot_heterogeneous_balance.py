from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise SystemExit(f"Missing {path}")
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"No rows in {path}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot uniform vs weighted heterogeneous MPI mapping.")
    parser.add_argument("--uniform-host-metrics", required=True)
    parser.add_argument("--weighted-host-metrics", required=True)
    parser.add_argument("--overview", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError as exc:
        raise SystemExit("Missing matplotlib. Install with: pip install -e '.[helpers]'") from exc

    uniform = read_rows(Path(args.uniform_host_metrics))
    weighted = read_rows(Path(args.weighted_host_metrics))
    overview = read_rows(Path(args.overview))

    host_order = sorted({row["hostname"] for row in uniform + weighted})

    def values(rows: list[dict[str, str]], key: str) -> list[float]:
        by_host = {row["hostname"]: float(row[key]) for row in rows}
        return [by_host.get(host, 0.0) for host in host_order]

    uniform_tasks = values(uniform, "task_share")
    weighted_tasks = values(weighted, "task_share")
    uniform_ranks = values(uniform, "rank_count")
    weighted_ranks = values(weighted, "rank_count")

    runtime_by_label = {
        row["label"]: float(row["total_ms_with_comm"]) / 1000.0
        for row in overview
    }
    imbalance_by_label = {
        row["label"]: float(row["load_imbalance"])
        for row in overview
    }

    x = list(range(len(host_order)))
    width = 0.36

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    ax.bar([i - width / 2 for i in x], uniform_tasks, width, label="uniform task share")
    ax.bar([i + width / 2 for i in x], weighted_tasks, width, label="weighted task share")
    for i, v in enumerate(uniform_ranks):
        ax.text(i - width / 2, uniform_tasks[i] + 0.015, f"{int(v)}r", ha="center", fontsize=8)
    for i, v in enumerate(weighted_ranks):
        ax.text(i + width / 2, weighted_tasks[i] + 0.015, f"{int(v)}r", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(host_order, rotation=20, ha="right")
    ax.set_ylabel("Task share")
    ax.set_title("Work distribution by host")
    ax.set_ylim(0, max(uniform_tasks + weighted_tasks + [0.1]) * 1.25)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="best")

    ax = axes[1]
    labels = ["uniform_24", "weighted_24"]
    runtimes = [runtime_by_label.get(label, 0.0) for label in labels]
    imbalances = [imbalance_by_label.get(label, 0.0) for label in labels]
    ax.bar(labels, runtimes, color=["#4c78a8", "#59a14f"])
    ax.set_ylabel("Runtime with communication (s)")
    ax.set_title("Runtime comparison")
    ax.grid(True, axis="y", alpha=0.25)

    ax2 = ax.twinx()
    ax2.plot(labels, imbalances, marker="o", color="#e15759", label="load imbalance")
    ax2.set_ylabel("Load imbalance")
    ax2.set_ylim(bottom=0)
    ax2.legend(loc="upper right")

    fig.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    print(f"HETEROGENEOUS_PLOT={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
