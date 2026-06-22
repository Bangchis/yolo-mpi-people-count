from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot runtime vs input size N from find_N.csv.")
    parser.add_argument("--input", required=True, help="find_N.csv")
    parser.add_argument("--output", required=True, help="Output PNG")
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError as exc:
        raise SystemExit("Missing matplotlib. Install with: pip install -e '.[helpers]'") from exc

    rows = []
    with Path(args.input).open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"No rows in {args.input}")

    frames = [int(float(r["frames"])) for r in rows]
    with_comm = [float(r["total_ms_with_comm"]) / 1000.0 for r in rows]
    without_comm = [float(r["total_ms_without_comm"]) / 1000.0 for r in rows]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(frames, with_comm, marker="o", label="with communication")
    ax.plot(frames, without_comm, marker="s", label="without communication")
    ax.axhline(120, color="0.55", linestyle="--", linewidth=1, label="2 min")
    ax.axhline(180, color="0.35", linestyle=":", linewidth=1, label="3 min")
    ax.set_xlabel("Input size N (frames)")
    ax.set_ylabel("Runtime (seconds)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    print(f"FIND_N_PLOT={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
