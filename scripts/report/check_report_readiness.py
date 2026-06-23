from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def check_file(path: Path, label: str, failures: list[str]) -> None:
    if not path.exists():
        failures.append(f"Missing {label}: {path}")
        return
    if path.stat().st_size == 0:
        failures.append(f"Empty {label}: {path}")


def pdf_page_count(path: Path) -> int:
    data = path.read_bytes()
    counts = [int(match.group(1)) for match in re.finditer(rb"/Count\s+(\d+)", data)]
    if counts:
        return max(counts)
    return len(re.findall(rb"/Type\s*/Page\b", data))


def main() -> int:
    parser = argparse.ArgumentParser(description="Check final report readiness before submission.")
    parser.add_argument("--report", default="reports/final_report_hust_style.md")
    parser.add_argument("--html", default="reports/final_report_hust_style.html")
    parser.add_argument("--pdf", default="")
    parser.add_argument("--summary", default="results/report_mot17_mini_final_20260623-154318/summary_tables.md")
    parser.add_argument("--allow-placeholders", action="store_true")
    args = parser.parse_args()

    failures: list[str] = []
    report = Path(args.report)
    html = Path(args.html)
    summary = Path(args.summary)
    if args.pdf:
        pdf = Path(args.pdf)
    else:
        final_pdf = Path("reports/final_report_hust_style.pdf")
        draft_pdf = Path("reports/final_report_hust_style_DRAFT.pdf")
        pdf = final_pdf if final_pdf.exists() else draft_pdf

    check_file(report, "Markdown report", failures)
    check_file(html, "HTML report", failures)
    check_file(pdf, "PDF report", failures)
    check_file(summary, "result summary", failures)
    check_file(Path("reports/final_submission_checklist.md"), "submission checklist", failures)
    check_file(Path("reports/requirement_coverage_audit.md"), "requirement audit", failures)
    check_file(Path("reports/defense_qa_notes.md"), "defense Q&A notes", failures)
    check_file(Path("results/report_mot17_mini_final_20260623-154318/speedup_2N/raw/speedup.csv"), "2N speedup CSV", failures)
    check_file(Path("results/report_mot17_mini_final_20260623-154318/speedup_2N/figures/speedup.png"), "2N speedup figure", failures)

    if report.exists():
        text = report.read_text(encoding="utf-8")
        placeholders = re.findall(r"<ID nhóm>|<Họ tên - MSSV>", text)
        if placeholders and not args.allow_placeholders:
            failures.append(f"Report still has placeholders: {', '.join(sorted(set(placeholders)))}")

        required_phrases = [
            "task-level parallelism",
            "hybrid temporal-spatial decomposition",
            "one-dimensional flattened mapping",
            "master-worker star topology",
            "Static scheduling",
            "six-hundred-frame workload",
            "twelve-hundred-frame workload",
            "1.939x",
        ]
        for phrase in required_phrases:
            if phrase not in text:
                failures.append(f"Missing expected report phrase: {phrase}")

        image_targets = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text)
        if len(image_targets) < 5:
            failures.append(f"Expected at least 5 figures, found {len(image_targets)}")
        for target in image_targets:
            image_path = (report.parent / target).resolve()
            if not image_path.exists():
                failures.append(f"Missing embedded image target: {target}")

    if html.exists():
        html_text = html.read_text(encoding="utf-8")
        html_placeholders = re.findall(r"&lt;ID nhóm&gt;|&lt;Họ tên - MSSV&gt;|<ID nhóm>|<Họ tên - MSSV>", html_text)
        if html_placeholders and not args.allow_placeholders:
            failures.append("HTML report still has identity placeholders")
        if html_text.count("data:image/") < 5:
            failures.append("HTML report does not appear to contain embedded figures")
        if html_text.count("<table>") < 8:
            failures.append("HTML report has fewer tables than expected")

    if pdf.exists():
        pages = pdf_page_count(pdf)
        if pages < 10 or pages > 20:
            failures.append(f"PDF page count must be 10-20 pages, found {pages}: {pdf}")

    if failures:
        print("REPORT_READY=NO")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("REPORT_READY=YES")
    print(f"REPORT={report}")
    print(f"HTML={html}")
    print(f"PDF={pdf}")
    print(f"SUMMARY={summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
