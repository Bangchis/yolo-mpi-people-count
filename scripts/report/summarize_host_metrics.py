from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate rank_metrics.csv by physical host.")
    parser.add_argument("--input", required=True, help="rank_metrics.csv")
    parser.add_argument("--output", required=True, help="host_metrics.csv")
    parser.add_argument("--label", default="")
    args = parser.parse_args()

    input_path = Path(args.input)
    rows = list(csv.DictReader(input_path.open(newline="", encoding="utf-8")))
    if not rows:
        raise SystemExit(f"No rows in {input_path}")

    grouped: dict[str, dict[str, float]] = defaultdict(lambda: {
        "rank_count": 0.0,
        "tasks_done": 0.0,
        "frames_done": 0.0,
        "compute_ms": 0.0,
        "comm_ms": 0.0,
        "idle_ms": 0.0,
        "yolo_ms": 0.0,
    })

    for row in rows:
        host = row["hostname"]
        g = grouped[host]
        g["rank_count"] += 1
        for key in ["tasks_done", "frames_done", "compute_ms", "comm_ms", "idle_ms", "yolo_ms"]:
            g[key] += float(row.get(key, 0.0) or 0.0)

    total_tasks = sum(g["tasks_done"] for g in grouped.values())
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "label",
            "hostname",
            "rank_count",
            "tasks_done",
            "task_share",
            "frames_done",
            "compute_s",
            "comm_s",
            "idle_s",
            "yolo_s",
        ])
        for host, g in sorted(grouped.items()):
            task_share = g["tasks_done"] / total_tasks if total_tasks > 0 else 0.0
            writer.writerow([
                args.label,
                host,
                int(g["rank_count"]),
                int(g["tasks_done"]),
                f"{task_share:.6f}",
                int(g["frames_done"]),
                f"{g['compute_ms'] / 1000.0:.6f}",
                f"{g['comm_ms'] / 1000.0:.6f}",
                f"{g['idle_ms'] / 1000.0:.6f}",
                f"{g['yolo_ms'] / 1000.0:.6f}",
            ])

    print(f"HOST_METRICS={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
