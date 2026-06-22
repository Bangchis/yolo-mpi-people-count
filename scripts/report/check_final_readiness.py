from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def add(rows: list[tuple[str, str, str]], status: str, check: str, detail: str) -> None:
    rows.append((status, check, detail))


def parse_hostfile(path: Path) -> list[tuple[str, int]]:
    hosts: list[tuple[str, int]] = []
    if not path.exists():
        return hosts
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        slots = 1
        for part in parts[1:]:
            if part.startswith("slots="):
                try:
                    slots = int(part.split("=", 1)[1])
                except ValueError:
                    slots = 0
        hosts.append((parts[0], slots))
    return hosts


def check_run_dir(run_dir: Path, rows: list[tuple[str, str, str]]) -> None:
    summary_path = run_dir / "summary.csv"
    summary_rows = read_csv(summary_path)
    if not summary_rows:
        add(rows, "FAIL", "summary.csv", f"missing or empty: {summary_path}")
        return
    summary = summary_rows[0]
    add(rows, "PASS", "summary.csv", str(summary_path))

    language = summary.get("language", "")
    detector = summary.get("detector", "")
    add(rows, "PASS" if language == "C++17/OpenMPI" else "FAIL", "parallel language", language)
    add(rows, "PASS" if detector == "yolo" else "WARN", "detector", detector)

    world_size = int(float(summary.get("world_size", "0") or 0))
    tasks = int(float(summary.get("num_tasks", "0") or 0))
    add(rows, "PASS" if world_size > 0 else "FAIL", "world_size", str(world_size))
    add(rows, "PASS" if tasks > 0 else "FAIL", "num_tasks", str(tasks))

    bboxes = read_csv(run_dir / "bboxes.csv")
    add(rows, "PASS" if bboxes else "WARN", "bboxes.csv", f"rows={len(bboxes)}")
    rank_metrics = read_csv(run_dir / "rank_metrics.csv")
    active_ranks = sorted({r.get("rank", "") for r in rank_metrics if int(float(r.get("tasks_done", "0") or 0)) > 0})
    add(rows, "PASS" if active_ranks else "FAIL", "active ranks", ",".join(active_ranks) or "none")
    if summary.get("master_compute", "0") in {"1", "true", "TRUE", "yes"}:
        add(rows, "PASS" if "0" in active_ranks else "FAIL", "master rank compute", ",".join(active_ranks) or "none")

    correctness_path = run_dir / "correctness.txt"
    if correctness_path.exists():
        text = correctness_path.read_text(encoding="utf-8", errors="ignore")
        add(rows, "PASS" if "CORRECTNESS_PASS=YES" in text else "FAIL", "correctness", correctness_path.name)
    else:
        add(rows, "WARN", "correctness", "missing correctness.txt")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--hostfile", default="configs/hosts_macos_gpu")
    parser.add_argument("--require-host", action="append", default=[])
    parser.add_argument("--require-yolo-model", default="models/yolo11n.pt")
    args = parser.parse_args()

    rows: list[tuple[str, str, str]] = []
    run_dir = Path(args.run_dir)
    add(rows, "PASS" if run_dir.exists() else "FAIL", "run dir", str(run_dir))
    add(rows, "PASS" if Path("build/yolo_mpi_cpp").exists() else "FAIL", "C++ binary", "build/yolo_mpi_cpp")
    add(rows, "PASS" if Path(args.require_yolo_model).exists() else "FAIL", "YOLO model", args.require_yolo_model)

    hostfile = Path(args.hostfile)
    hosts = parse_hostfile(hostfile)
    add(rows, "PASS" if hosts else "WARN", "hostfile", str(hostfile))
    host_names = [host for host, _ in hosts]
    for required in args.require_host:
        add(rows, "PASS" if required in host_names else "FAIL", f"required host {required}", ",".join(host_names))
    if hosts:
        total_slots = sum(slots for _, slots in hosts)
        add(rows, "PASS" if total_slots > 0 else "FAIL", "host slots", str(total_slots))

    if run_dir.exists():
        check_run_dir(run_dir, rows)

    output = run_dir / "readiness.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["status", "check", "detail"])
        writer.writerows(rows)

    for status, check, detail in rows:
        print(f"{status}\t{check}\t{detail}")
    print(f"READINESS_CSV={output}")

    failed = [row for row in rows if row[0] == "FAIL"]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
