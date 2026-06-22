from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot ground-truth vs predicted counts and absolute error.")
    parser.add_argument("--input", required=True, help="per_frame_accuracy.csv")
    parser.add_argument("--output", required=True, help="Output PNG")
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError as exc:
        raise SystemExit("Missing matplotlib. Install with: pip install -e '.[helpers]'") from exc

    with Path(args.input).open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"No rows in {args.input}")

    frame_id = [int(r["frame_id"]) for r in rows]
    gt = [int(float(r["gt_count"])) for r in rows]
    pred = [int(float(r["pred_count"])) for r in rows]
    abs_error = [float(r["abs_error"]) for r in rows]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    ax1.plot(frame_id, gt, label="ground truth", linewidth=1.6)
    ax1.plot(frame_id, pred, label="YOLO prediction", linewidth=1.4)
    ax1.set_ylabel("People count")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="best")

    ax2.bar(frame_id, abs_error, width=1.0, color="#d95f02")
    ax2.set_xlabel("Frame")
    ax2.set_ylabel("Abs error")
    ax2.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    print(f"COUNT_ERROR_PLOT={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
