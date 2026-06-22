from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="speedup.csv from run_speedup_sweep.sh")
    parser.add_argument("--output", required=True, help="output PNG")
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError as exc:
        raise SystemExit("Missing matplotlib. Install with: pip install -e '.[helpers]'") from exc

    rows = []
    with Path(args.input).open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    if not rows:
        raise SystemExit(f"No rows in {args.input}")

    p = [int(r["world_size"]) for r in rows]
    speedup_with = [float(r["speedup_with_comm"]) for r in rows]
    speedup_without = [float(r["speedup_without_comm"]) for r in rows]
    efficiency_with = [float(r["efficiency_with_comm"]) for r in rows]

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(p, speedup_with, marker="o", label="speedup with comm")
    ax1.plot(p, speedup_without, marker="s", label="speedup without comm")
    ax1.plot(p, p, linestyle="--", color="0.65", label="ideal")
    ax1.set_xlabel("MPI processes")
    ax1.set_ylabel("Speedup")
    ax1.grid(True, alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(p, efficiency_with, marker="^", color="#d95f02", label="efficiency with comm")
    ax2.set_ylabel("Efficiency")
    ax2.set_ylim(bottom=0)

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="best")
    fig.tight_layout()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    print(f"SPEEDUP_PLOT={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
