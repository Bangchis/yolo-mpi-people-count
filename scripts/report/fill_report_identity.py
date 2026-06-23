from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Fill group identity placeholders in the final report.")
    parser.add_argument("--group-id", required=True, help="Real group ID")
    parser.add_argument(
        "--members",
        required=True,
        help="Member list, for example: 'Nguyen Van A - 20230001; Tran Van B - 20230002'",
    )
    parser.add_argument("--input", default="reports/final_report_hust_style.md")
    parser.add_argument("--output", default="reports/final_report_hust_style.md")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    text = input_path.read_text(encoding="utf-8")
    text = text.replace("<ID nhóm>", args.group_id)
    text = text.replace("<Họ tên - MSSV>", args.members)

    output_path.write_text(text, encoding="utf-8")
    print(f"REPORT_IDENTITY_FILLED={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
