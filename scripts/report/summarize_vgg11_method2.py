#!/usr/bin/env python3
"""Summarize Method 2 VGG11 distributed convolution results for the report."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_manifest(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values

    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def md_table(rows: list[dict[str, str]], columns: list[tuple[str, str]]) -> str:
    if not rows:
        return "_No data available yet._\n"

    headers = [label for _, label in columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]

    for row in rows:
        lines.append("| " + " | ".join(row.get(key, "") for key, _ in columns) + " |")

    return "\n".join(lines) + "\n"


def fmt_float(value: str, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return value


def summarize_input_size(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out = []
    for row in rows:
        out.append({
            "size": f"{row.get('height', '')}x{row.get('width', '')}",
            "p": row.get("p", ""),
            "halo_mode": row.get("halo_mode", ""),
            "runtime_s": fmt_float(str(float(row.get("distributed_ms", "0")) / 1000.0), 4),
            "halo_ms": fmt_float(row.get("halo_ms", "0"), 3),
            "compute_ms": fmt_float(row.get("compute_ms", "0"), 3),
            "correct": row.get("correct", ""),
        })
    return out


def summarize_speedup(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out = []
    for row in rows:
        out.append({
            "p": row.get("p", ""),
            "halo_mode": row.get("halo_mode", ""),
            "runtime_s": fmt_float(str(float(row.get("distributed_ms", "0")) / 1000.0), 4),
            "speedup": fmt_float(row.get("speedup", "0"), 3),
            "efficiency": fmt_float(row.get("efficiency", "0"), 3),
            "inter_ratio": fmt_float(row.get("halo_inter_machine_edge_ratio", "0"), 3),
            "correct": row.get("correct", ""),
        })
    return out


def summarize_topology(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out = []
    for row in rows:
        out.append({
            "label": row.get("label", ""),
            "runtime_s": fmt_float(str(float(row.get("runtime_ms", "0")) / 1000.0), 4),
            "grid": row.get("grid", ""),
            "inter_edges": row.get("inter_machine_edges", ""),
            "inter_ratio": fmt_float(row.get("inter_machine_edge_ratio", "0"), 3),
            "correct": row.get("correct", ""),
        })
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    run_dir = args.run_dir
    output = args.output or run_dir / "summary_tables.md"

    manifest = read_manifest(run_dir / "manifest.txt")
    input_rows = read_csv(run_dir / "input_size" / "input_size.csv")
    speedup_rows = read_csv(run_dir / "speedup" / "raw" / "vgg11_speedup.csv")
    topology_rows = read_csv(run_dir / "topology" / "topology_mapping_comparison.csv")

    lines: list[str] = []
    lines.append("# Method 2 VGG11 Distributed Convolution Summary\n")
    lines.append("## Configuration\n")
    config_rows = [{"key": k, "value": v} for k, v in sorted(manifest.items())]
    lines.append(md_table(config_rows, [("key", "Key"), ("value", "Value")]))

    lines.append("## Correctness Status\n")
    all_rows = input_rows + speedup_rows + topology_rows
    correct_values = sorted({row.get("correct", "") for row in all_rows if row.get("correct", "")})
    max_error = 0.0
    for row in all_rows:
        try:
            max_error = max(max_error, float(row.get("max_abs_error", "0")))
        except ValueError:
            pass
    lines.append(f"- Correctness values observed: `{', '.join(correct_values) if correct_values else 'N/A'}`\n")
    lines.append(f"- Maximum absolute error observed: `{max_error:.8f}`\n")

    lines.append("## Input Size Selection\n")
    lines.append(md_table(
        summarize_input_size(input_rows),
        [
            ("size", "Size"),
            ("p", "P"),
            ("halo_mode", "Halo Mode"),
            ("runtime_s", "Runtime (s)"),
            ("halo_ms", "Halo ms"),
            ("compute_ms", "Compute ms"),
            ("correct", "Correct"),
        ],
    ))

    lines.append("## Speedup and Efficiency\n")
    lines.append(md_table(
        summarize_speedup(speedup_rows),
        [
            ("p", "P"),
            ("halo_mode", "Halo Mode"),
            ("runtime_s", "Runtime (s)"),
            ("speedup", "Speedup"),
            ("efficiency", "Efficiency"),
            ("inter_ratio", "Inter-machine Halo Ratio"),
            ("correct", "Correct"),
        ],
    ))

    lines.append("## Topology-Aware Mapping\n")
    if topology_rows:
        lines.append(md_table(
            summarize_topology(topology_rows),
            [
                ("label", "Mapping"),
                ("runtime_s", "Runtime (s)"),
                ("grid", "Grid"),
                ("inter_edges", "Inter-machine Edges"),
                ("inter_ratio", "Inter-machine Ratio"),
                ("correct", "Correct"),
            ],
        ))
    else:
        lines.append("_Topology comparison not available. Run the report suite on the three-machine LAN with `VGG_RUN_TOPOLOGY=1`._\n")

    lines.append("## Figures\n")
    lines.append("- `figures/vgg11_input_size.png`\n")
    lines.append("- `speedup/figures/vgg11_conv_method2.png`\n")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"METHOD2_SUMMARY={output}")


if __name__ == "__main__":
    main()
