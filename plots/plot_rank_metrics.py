from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_metrics(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"No rows in {path}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot rank compute/communication/idle stacked bars.")
    parser.add_argument("--input", required=True, help="rank_metrics.csv")
    parser.add_argument("--output", required=True, help="Output PNG")
    parser.add_argument("--summary-output", help="One-row granularity summary CSV")
    parser.add_argument("--label", default="", help="Granularity label, for example grid_4x3")
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError as exc:
        raise SystemExit("Missing matplotlib. Install with: pip install -e '.[helpers]'") from exc

    rows = read_metrics(Path(args.input))
    ranks = [int(r["rank"]) for r in rows]
    compute = [float(r.get("compute_ms", 0.0)) / 1000.0 for r in rows]
    comm = [float(r.get("comm_ms", 0.0)) / 1000.0 for r in rows]
    idle = [float(r.get("idle_ms", 0.0)) / 1000.0 for r in rows]
    tasks = [int(float(r.get("tasks_done", 0))) for r in rows]

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(ranks, compute, label="compute")
    ax.bar(ranks, comm, bottom=compute, label="communication")
    bottom = [c + m for c, m in zip(compute, comm)]
    ax.bar(ranks, idle, bottom=bottom, label="idle")
    ax.set_xlabel("MPI rank")
    ax.set_ylabel("Time (seconds)")
    ax.set_xticks(ranks)
    ax.grid(True, axis="y", alpha=0.25)
    title = "Rank time breakdown"
    if args.label:
        title += f" ({args.label})"
    ax.set_title(title)
    ax.legend(loc="best")
    fig.tight_layout()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    print(f"RANK_METRICS_PLOT={output}")

    if args.summary_output:
        active_compute = [v for v, task_count in zip(compute, tasks) if task_count > 0]
        active_idle = [v for v, task_count in zip(idle, tasks) if task_count > 0]
        compute_max = max(active_compute) if active_compute else 0.0
        compute_min = min(active_compute) if active_compute else 0.0
        compute_avg = sum(active_compute) / len(active_compute) if active_compute else 0.0
        idle_max = max(active_idle) if active_idle else 0.0
        idle_min = min(active_idle) if active_idle else 0.0
        idle_gap_ratio = (idle_max - idle_min) / compute_max if compute_max > 0 else 0.0
        load_balance_pass = "YES" if idle_gap_ratio <= 0.25 else "NO"
        summary_output = Path(args.summary_output)
        summary_output.parent.mkdir(parents=True, exist_ok=True)
        with summary_output.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "label",
                    "rank_count",
                    "total_tasks",
                    "compute_s_max",
                    "compute_s_min",
                    "compute_s_avg",
                    "comm_s_total",
                    "idle_s_total",
                    "idle_gap_ratio",
                    "load_balance_pass",
                ]
            )
            writer.writerow(
                [
                    args.label,
                    len(rows),
                    sum(tasks),
                    f"{compute_max:.6f}",
                    f"{compute_min:.6f}",
                    f"{compute_avg:.6f}",
                    f"{sum(comm):.6f}",
                    f"{sum(idle):.6f}",
                    f"{idle_gap_ratio:.6f}",
                    load_balance_pass,
                ]
            )
        print(f"GRANULARITY_SUMMARY={summary_output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
