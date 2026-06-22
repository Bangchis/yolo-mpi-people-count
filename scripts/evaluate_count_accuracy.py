from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def read_counts(path: Path, preferred_fields: tuple[str, ...]) -> dict[int, int]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise SystemExit(f"No CSV header in {path}")
        count_field = next((name for name in preferred_fields if name in reader.fieldnames), None)
        if count_field is None:
            count_field = "person_count" if "person_count" in reader.fieldnames else None
        if count_field is None:
            raise SystemExit(f"{path} must contain one of {preferred_fields} or person_count")
        if "frame_id" not in reader.fieldnames:
            raise SystemExit(f"{path} must contain frame_id")
        rows: dict[int, int] = {}
        for row in reader:
            rows[int(row["frame_id"])] = int(round(float(row[count_field])))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate predicted people counts against ground truth counts.")
    parser.add_argument("--predicted", required=True, help="Predicted frame_counts.csv")
    parser.add_argument("--ground-truth", required=True, help="Ground truth counts CSV")
    parser.add_argument("--summary-output", required=True, help="Summary accuracy CSV")
    parser.add_argument("--per-frame-output", required=True, help="Per-frame accuracy CSV")
    args = parser.parse_args()

    predicted_path = Path(args.predicted)
    gt_path = Path(args.ground_truth)
    pred = read_counts(predicted_path, ("pred_count", "person_count", "count"))
    gt = read_counts(gt_path, ("gt_count", "person_count", "count"))

    frames = sorted(set(gt) & set(pred))
    if not frames:
        raise SystemExit("No overlapping frame_id values between prediction and ground truth")

    per_frame_rows = []
    abs_errors = []
    sq_errors = []
    ape_values = []
    exact = 0
    gt_sum = 0
    pred_sum = 0
    for frame_id in frames:
        gt_count = gt[frame_id]
        pred_count = pred[frame_id]
        error = pred_count - gt_count
        abs_error = abs(error)
        sq_error = error * error
        ape = "" if gt_count == 0 else abs_error / gt_count
        if ape != "":
            ape_values.append(float(ape))
        abs_errors.append(abs_error)
        sq_errors.append(sq_error)
        exact += 1 if error == 0 else 0
        gt_sum += gt_count
        pred_sum += pred_count
        per_frame_rows.append(
            {
                "frame_id": frame_id,
                "gt_count": gt_count,
                "pred_count": pred_count,
                "error": error,
                "abs_error": abs_error,
                "squared_error": sq_error,
                "absolute_percentage_error": "" if ape == "" else f"{ape:.6f}",
            }
        )

    n = len(frames)
    mae = sum(abs_errors) / n
    rmse = math.sqrt(sum(sq_errors) / n)
    mape = sum(ape_values) / len(ape_values) if ape_values else 0.0
    exact_match_rate = exact / n

    per_frame_output = Path(args.per_frame_output)
    per_frame_output.parent.mkdir(parents=True, exist_ok=True)
    with per_frame_output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "frame_id",
                "gt_count",
                "pred_count",
                "error",
                "abs_error",
                "squared_error",
                "absolute_percentage_error",
            ],
        )
        writer.writeheader()
        writer.writerows(per_frame_rows)

    summary_output = Path(args.summary_output)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    with summary_output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "frames_compared",
                "mae",
                "rmse",
                "mean_abs_percentage_error",
                "exact_match_rate",
                "mean_gt_count",
                "mean_pred_count",
                "predicted_csv",
                "ground_truth_csv",
            ]
        )
        writer.writerow(
            [
                n,
                f"{mae:.6f}",
                f"{rmse:.6f}",
                f"{mape:.6f}",
                f"{exact_match_rate:.6f}",
                f"{gt_sum / n:.6f}",
                f"{pred_sum / n:.6f}",
                predicted_path,
                gt_path,
            ]
        )

    print(f"ACCURACY_SUMMARY={summary_output}")
    print(f"ACCURACY_PER_FRAME={per_frame_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
