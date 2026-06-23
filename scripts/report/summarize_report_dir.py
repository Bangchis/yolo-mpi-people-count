from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_first_csv(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[0] if rows else None


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fmt_float(value: str | float, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    out = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def existing_figures(report_dir: Path) -> list[tuple[str, Path]]:
    """Return the important report figures that currently exist."""
    candidates = [
        ("Count error by frame", report_dir / "accuracy" / "count_error_plot.png"),
        ("Find N runtime", report_dir / "find_N" / "figures" / "find_N_runtime.png"),
        ("Granularity overview", report_dir / "granularity" / "granularity_overview.png"),
        ("Static vs dynamic scheduler", report_dir / "scheduler" / "figures" / "scheduler_comparison.png"),
        ("Speedup", report_dir / "speedup" / "figures" / "speedup.png"),
        ("Speedup 2N", report_dir / "speedup_2N" / "figures" / "speedup.png"),
        ("Heterogeneous weighted mapping", report_dir / "heterogeneous" / "figures" / "heterogeneous_balance.png"),
    ]

    granularity_dir = report_dir / "granularity"
    if granularity_dir.exists():
        for path in sorted(granularity_dir.glob("grid_*/rank_metrics_stacked.png")):
            candidates.append((f"Granularity {path.parent.name}", path))

    for path in sorted(report_dir.glob("find_N_long*/figures/find_N_runtime.png")):
        candidates.append((f"Long Find N {path.parent.parent.name}", path))

    return [(label, path) for label, path in candidates if path.exists()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Create Markdown tables from a report result directory.")
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    report_dir = Path(args.report_dir)
    if not report_dir.exists():
        raise SystemExit(f"Missing report directory: {report_dir}")

    lines: list[str] = []
    lines.append(f"# Report Result Summary: `{report_dir}`")
    lines.append("")

    manifest = report_dir / "manifest.txt"
    if manifest.exists():
        lines.append("## Manifest")
        lines.append("")
        lines.append("```text")
        lines.append(manifest.read_text(encoding="utf-8").strip())
        lines.append("```")
        lines.append("")

    correctness = read_first_csv(report_dir / "correctness" / "correctness_compare.csv")
    if correctness:
        lines.append("## Correctness: Serial vs MPI")
        lines.append("")
        lines.append(
            md_table(
                ["Pass", "Frames", "Mismatched", "Max Error", "Mean Error"],
                [[
                    correctness.get("correctness_pass", ""),
                    correctness.get("frames_compared", ""),
                    correctness.get("mismatched_frames", ""),
                    correctness.get("max_abs_error", ""),
                    fmt_float(correctness.get("mean_abs_error", "")),
                ]],
            )
        )
        lines.append("")

    accuracy = read_first_csv(report_dir / "accuracy" / "accuracy.csv")
    if accuracy:
        lines.append("## YOLO Count Accuracy vs MOT Ground Truth")
        lines.append("")
        lines.append(
            md_table(
                ["Frames", "MAE", "RMSE", "MAPE", "Exact", "GT Avg", "Pred Avg"],
                [[
                    accuracy.get("frames_compared", ""),
                    fmt_float(accuracy.get("mae", "")),
                    fmt_float(accuracy.get("rmse", "")),
                    fmt_float(accuracy.get("mean_abs_percentage_error", "")),
                    fmt_float(accuracy.get("exact_match_rate", "")),
                    fmt_float(accuracy.get("mean_gt_count", "")),
                    fmt_float(accuracy.get("mean_pred_count", "")),
                ]],
            )
        )
        lines.append("")

    find_n_rows = read_csv(report_dir / "find_N" / "raw" / "find_N.csv")
    if find_n_rows:
        lines.append("## Find N")
        lines.append("")
        lines.append(
            md_table(
                ["Frames", "With Comm (s)", "Without Comm (s)", "P", "Grid"],
                [[
                    row.get("frames", ""),
                    fmt_float(float(row.get("total_ms_with_comm", "0")) / 1000.0),
                    fmt_float(float(row.get("total_ms_without_comm", "0")) / 1000.0),
                    row.get("world_size", ""),
                    row.get("tile_grid", ""),
                ] for row in find_n_rows],
            )
        )
        lines.append("")

    for long_find_n in sorted(report_dir.glob("find_N_long*/raw/find_N.csv")):
        long_rows = read_csv(long_find_n)
        if not long_rows:
            continue
        lines.append(f"## Long Find N: {long_find_n.parent.parent.name}")
        lines.append("")
        lines.append(
            md_table(
                ["Frames", "With Comm (s)", "Without Comm (s)", "P", "Grid"],
                [[
                    row.get("frames", ""),
                    fmt_float(float(row.get("total_ms_with_comm", "0")) / 1000.0),
                    fmt_float(float(row.get("total_ms_without_comm", "0")) / 1000.0),
                    row.get("world_size", ""),
                    row.get("tile_grid", ""),
                ] for row in long_rows],
            )
        )
        lines.append("")

    granularity_rows = read_csv(report_dir / "granularity" / "granularity_overview.csv")
    if granularity_rows:
        lines.append("## Granularity And Load Balance")
        lines.append("")
        lines.append(
            md_table(
                ["Grid", "Ranks", "Tasks", "Compute Max (s)", "Compute Avg (s)", "Comm Total (s)", "Idle Total (s)", "Idle Gap", "Pass"],
                [[
                    row.get("label", ""),
                    row.get("rank_count", ""),
                    row.get("total_tasks", ""),
                    fmt_float(row.get("compute_s_max", "")),
                    fmt_float(row.get("compute_s_avg", "")),
                    fmt_float(row.get("comm_s_total", "")),
                    fmt_float(row.get("idle_s_total", "")),
                    fmt_float(row.get("idle_gap_ratio", "")),
                    row.get("load_balance_pass", ""),
                ] for row in granularity_rows],
            )
        )
        lines.append("")

    scheduler_rows = read_csv(report_dir / "scheduler" / "scheduler_comparison.csv")
    if scheduler_rows:
        lines.append("## Scheduler Comparison")
        lines.append("")
        lines.append(
            md_table(
                ["Schedule", "P", "Frames", "Grid", "With Comm (s)", "Without Comm (s)", "Load Imbalance", "Idle Gap", "Pass"],
                [[
                    row.get("schedule", ""),
                    row.get("world_size", ""),
                    row.get("frames", ""),
                    row.get("tile_grid", ""),
                    fmt_float(float(row.get("total_ms_with_comm", "0")) / 1000.0),
                    fmt_float(float(row.get("total_ms_without_comm", "0")) / 1000.0),
                    fmt_float(row.get("load_imbalance", "")),
                    fmt_float(row.get("idle_gap_ratio", "")),
                    row.get("load_balance_pass", ""),
                ] for row in scheduler_rows],
            )
        )
        lines.append("")

    speedup_rows = read_csv(report_dir / "speedup" / "raw" / "speedup.csv")
    if speedup_rows:
        lines.append("## Speedup")
        lines.append("")
        lines.append(
            md_table(
                ["P", "With Comm (s)", "Without Comm (s)", "Speedup", "Efficiency"],
                [[
                    row.get("world_size", ""),
                    fmt_float(float(row.get("total_ms_with_comm", "0")) / 1000.0),
                    fmt_float(float(row.get("total_ms_without_comm", "0")) / 1000.0),
                    fmt_float(row.get("speedup_with_comm", "")),
                    fmt_float(row.get("efficiency_with_comm", "")),
                ] for row in speedup_rows],
            )
        )
        lines.append("")

    speedup_2n_rows = read_csv(report_dir / "speedup_2N" / "raw" / "speedup.csv")
    if speedup_2n_rows:
        lines.append("## Speedup 2N")
        lines.append("")
        lines.append(
            md_table(
                ["P", "With Comm (s)", "Without Comm (s)", "Speedup", "Efficiency"],
                [[
                    row.get("world_size", ""),
                    fmt_float(float(row.get("total_ms_with_comm", "0")) / 1000.0),
                    fmt_float(float(row.get("total_ms_without_comm", "0")) / 1000.0),
                    fmt_float(row.get("speedup_with_comm", "")),
                    fmt_float(row.get("efficiency_with_comm", "")),
                ] for row in speedup_2n_rows],
            )
        )
        lines.append("")

    heterogeneous_rows = read_csv(report_dir / "heterogeneous" / "heterogeneous_overview.csv")
    if heterogeneous_rows:
        lines.append("## Heterogeneous Weighted Mapping")
        lines.append("")
        lines.append(
            md_table(
                ["Case", "P", "Frames", "Grid", "With Comm (s)", "Without Comm (s)", "Load Imbalance", "Host Task Distribution"],
                [[
                    row.get("label", ""),
                    row.get("world_size", ""),
                    row.get("frames", ""),
                    row.get("tile_grid", ""),
                    fmt_float(float(row.get("total_ms_with_comm", "0")) / 1000.0),
                    fmt_float(float(row.get("total_ms_without_comm", "0")) / 1000.0),
                    fmt_float(row.get("load_imbalance", "")),
                    row.get("host_task_distribution", "").replace("|", "/"),
                ] for row in heterogeneous_rows],
            )
        )
        lines.append("")

    figures = existing_figures(report_dir)
    if figures:
        lines.append("## Figures To Include")
        lines.append("")
        lines.append(
            md_table(
                ["Purpose", "File"],
                [[label, str(path)] for label, path in figures],
            )
        )
        lines.append("")

    output_text = "\n".join(lines).rstrip() + "\n"
    output = Path(args.output) if args.output else report_dir / "summary_tables.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(output_text, encoding="utf-8")
    print(f"REPORT_SUMMARY_TABLES={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
