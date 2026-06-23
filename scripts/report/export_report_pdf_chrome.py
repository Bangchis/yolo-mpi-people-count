from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def find_chrome() -> Path | None:
    candidates = [
        Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
        Path("/Applications/Chromium.app/Contents/MacOS/Chromium"),
        Path("/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    for name in ["google-chrome", "chromium", "chromium-browser"]:
        resolved = shutil.which(name)
        if resolved:
            return Path(resolved)
    return None


def page_count(pdf_path: Path) -> int:
    data = pdf_path.read_bytes()
    counts = [int(match.group(1)) for match in re.finditer(rb"/Count\s+(\d+)", data)]
    if counts:
        return max(counts)
    return len(re.findall(rb"/Type\s*/Page\b", data))


def wait_for_pdf(pdf_path: Path, timeout_s: float) -> None:
    start = time.monotonic()
    last_size = -1
    stable_count = 0

    while time.monotonic() - start < timeout_s:
        if pdf_path.exists():
            size = pdf_path.stat().st_size
            if size > 0 and size == last_size:
                stable_count += 1
            else:
                stable_count = 0
            last_size = size

            if stable_count >= 3:
                return

        time.sleep(0.5)

    raise TimeoutError(f"Timed out waiting for PDF output: {pdf_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Export the report HTML to PDF with headless Chrome.")
    parser.add_argument("--input-html", default="reports/final_report_hust_style.html")
    parser.add_argument("--output-pdf", default="reports/final_report_hust_style.pdf")
    parser.add_argument("--min-pages", type=int, default=10)
    parser.add_argument("--max-pages", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    html_path = Path(args.input_html).resolve()
    pdf_path = Path(args.output_pdf).resolve()

    if not html_path.exists():
        print(f"Missing HTML input: {html_path}", file=sys.stderr)
        return 1

    chrome = find_chrome()
    if chrome is None:
        print("Could not find Chrome/Chromium/Edge for PDF export.", file=sys.stderr)
        return 1

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    if pdf_path.exists():
        pdf_path.unlink()

    profile_dir = Path(tempfile.mkdtemp(prefix="chrome-yolo-report-"))
    command = [
        str(chrome),
        "--headless=new",
        "--disable-gpu",
        "--no-first-run",
        "--disable-background-networking",
        f"--user-data-dir={profile_dir}",
        f"--print-to-pdf={pdf_path}",
        html_path.as_uri(),
    ]

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        wait_for_pdf(pdf_path, args.timeout)
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.communicate(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()

    if not pdf_path.exists() or pdf_path.stat().st_size == 0:
        print(f"PDF was not created: {pdf_path}", file=sys.stderr)
        return 1

    pages = page_count(pdf_path)
    print(f"REPORT_PDF={pdf_path}")
    print(f"PDF_PAGES={pages}")
    print(f"PDF_BYTES={pdf_path.stat().st_size}")

    if pages < args.min_pages or pages > args.max_pages:
        print(
            f"PDF page count {pages} is outside required range "
            f"{args.min_pages}-{args.max_pages}.",
            file=sys.stderr,
        )
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
