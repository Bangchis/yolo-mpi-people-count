from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_counts(path: Path) -> dict[int, int]:
    rows: dict[int, int] = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise SystemExit(f"No CSV header in {path}")
        count_field = "person_count" if "person_count" in reader.fieldnames else "count"
        if count_field not in reader.fieldnames:
            raise SystemExit(f"{path} must contain person_count or count")
        for row in reader:
            rows[int(row["frame_id"])] = int(round(float(row[count_field])))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare serial and MPI frame count CSV files.")
    parser.add_argument("--serial", required=True, help="Serial frame_counts.csv")
    parser.add_argument("--mpi", required=True, help="MPI frame_counts.csv")
    parser.add_argument("--output", required=True, help="Summary CSV output")
    parser.add_argument("--per-frame-output", help="Optional per-frame comparison CSV")
    args = parser.parse_args()

    serial_path = Path(args.serial)
    mpi_path = Path(args.mpi)
    serial = read_counts(serial_path)
    mpi = read_counts(mpi_path)

    frames = sorted(set(serial) | set(mpi))
    if not frames:
        raise SystemExit("No frame counts to compare")

    rows = []
    max_abs_error = 0
    total_abs_error = 0
    mismatches = 0
    for frame_id in frames:
        s = serial.get(frame_id)
        m = mpi.get(frame_id)
        if s is None or m is None:
            error = 1
            abs_error = 1
        else:
            error = m - s
            abs_error = abs(error)
        max_abs_error = max(max_abs_error, abs_error)
        total_abs_error += abs_error
        if abs_error != 0:
            mismatches += 1
        rows.append(
            {
                "frame_id": frame_id,
                "serial_count": "" if s is None else s,
                "mpi_count": "" if m is None else m,
                "error": error,
                "abs_error": abs_error,
                "match": 1 if abs_error == 0 else 0,
            }
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "correctness_pass",
                "frames_compared",
                "mismatched_frames",
                "max_abs_error",
                "mean_abs_error",
                "serial_csv",
                "mpi_csv",
            ]
        )
        writer.writerow(
            [
                "YES" if mismatches == 0 else "NO",
                len(frames),
                mismatches,
                max_abs_error,
                f"{total_abs_error / len(frames):.6f}",
                serial_path,
                mpi_path,
            ]
        )

    if args.per_frame_output:
        per_frame_output = Path(args.per_frame_output)
        per_frame_output.parent.mkdir(parents=True, exist_ok=True)
        with per_frame_output.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["frame_id", "serial_count", "mpi_count", "error", "abs_error", "match"],
            )
            writer.writeheader()
            writer.writerows(rows)

    print(f"CORRECTNESS_COMPARE={output}")
    return 0 if mismatches == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
