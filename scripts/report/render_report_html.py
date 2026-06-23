from __future__ import annotations

import argparse
import base64
import html
import re
from pathlib import Path


def inline(text: str) -> str:
    text = html.escape(text)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    return text


def image_data_uri(markdown_path: Path, target: str) -> str:
    path = (markdown_path.parent / target).resolve()
    suffix = path.suffix.lower()
    mime = "image/png"
    if suffix in {".jpg", ".jpeg"}:
        mime = "image/jpeg"
    elif suffix == ".svg":
        mime = "image/svg+xml"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def render_table(lines: list[str]) -> str:
    rows = []
    for line in lines:
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        rows.append(cells)
    if len(rows) < 2:
        return ""

    header = rows[0]
    body = rows[2:]
    out = ["<table>", "<thead><tr>"]
    for cell in header:
        out.append(f"<th>{inline(cell)}</th>")
    out.append("</tr></thead>")
    out.append("<tbody>")
    for row in body:
        out.append("<tr>")
        for cell in row:
            out.append(f"<td>{inline(cell)}</td>")
        out.append("</tr>")
    out.append("</tbody></table>")
    return "\n".join(out)


def render_markdown(markdown_path: Path) -> str:
    lines = markdown_path.read_text(encoding="utf-8").splitlines()
    out: list[str] = ["<section class=\"cover-page\">"]
    paragraph: list[str] = []
    in_code = False
    code_lines: list[str] = []
    list_open = False
    cover_open = True
    ordered_list_open = False

    def flush_paragraph() -> None:
        nonlocal paragraph
        if paragraph:
            out.append(f"<p>{inline(' '.join(paragraph))}</p>")
            paragraph = []

    def close_list() -> None:
        nonlocal list_open, ordered_list_open
        if list_open:
            out.append("</ul>")
            list_open = False
        if ordered_list_open:
            out.append("</ol>")
            ordered_list_open = False

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith("```"):
            if in_code:
                out.append("<pre><code>" + html.escape("\n".join(code_lines)) + "</code></pre>")
                code_lines = []
                in_code = False
            else:
                flush_paragraph()
                close_list()
                in_code = True
            i += 1
            continue

        if in_code:
            code_lines.append(line)
            i += 1
            continue

        if not stripped:
            flush_paragraph()
            close_list()
            i += 1
            continue

        if stripped == "---":
            flush_paragraph()
            close_list()
            if cover_open:
                out.append("</section>")
                cover_open = False
            else:
                out.append("<hr>")
            i += 1
            continue

        if stripped.startswith("|") and i + 1 < len(lines) and lines[i + 1].strip().startswith("|"):
            flush_paragraph()
            close_list()
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i])
                i += 1
            out.append(render_table(table_lines))
            continue

        image_match = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", stripped)
        if image_match:
            flush_paragraph()
            close_list()
            alt, target = image_match.groups()
            uri = image_data_uri(markdown_path, target)
            out.append(
                "<figure>"
                f"<img src=\"{uri}\" alt=\"{html.escape(alt)}\">"
                f"<figcaption>{inline(alt)}</figcaption>"
                "</figure>"
            )
            i += 1
            continue

        heading_match = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if heading_match:
            flush_paragraph()
            close_list()
            level = len(heading_match.group(1))
            title = heading_match.group(2)
            attrs = ""
            if level == 2 and re.match(r"(\d+\.|Phụ Lục)", title):
                attrs = ' class="chapter"'
            out.append(f"<h{level}{attrs}>{inline(title)}</h{level}>")
            i += 1
            continue

        if stripped.startswith("- "):
            flush_paragraph()
            if ordered_list_open:
                out.append("</ol>")
                ordered_list_open = False
            if not list_open:
                out.append("<ul>")
                list_open = True
            out.append(f"<li>{inline(stripped[2:])}</li>")
            i += 1
            continue

        ordered_match = re.match(r"^\d+\.\s+(.*)$", stripped)
        if ordered_match:
            flush_paragraph()
            if list_open:
                out.append("</ul>")
                list_open = False
            if not ordered_list_open:
                out.append("<ol>")
                ordered_list_open = True
            out.append(f"<li>{inline(ordered_match.group(1))}</li>")
            i += 1
            continue

        paragraph.append(stripped)
        i += 1

    flush_paragraph()
    close_list()
    if cover_open:
        out.append("</section>")
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description="Render the final Markdown report into self-contained HTML.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    input_path = Path(args.input)
    body = render_markdown(input_path)

    css = """
    @page { size: A4; margin: 20mm 20mm 25mm 30mm; }
    body {
      font-family: "Times New Roman", Times, serif;
      color: #000000;
      line-height: 1.30;
      font-size: 13pt;
      max-width: 780px;
      margin: 0 auto;
      background: #ffffff;
    }
    .cover-page {
      min-height: 94vh;
      display: flex;
      flex-direction: column;
      justify-content: center;
      text-align: center;
      break-after: page;
      page-break-after: always;
    }
    .cover-page h1 {
      font-size: 17pt;
      letter-spacing: 0.02em;
      margin: 0 0 12pt;
      text-transform: uppercase;
    }
    .cover-page h2 {
      font-size: 18pt;
      border: 0;
      margin: 28pt 0 24pt;
      text-transform: uppercase;
    }
    .cover-page h3 {
      font-size: 13pt;
      margin: 4pt 0;
      text-transform: uppercase;
    }
    .cover-page p {
      text-align: center;
      margin: 5pt 0;
    }
    h1, h2, h3 { color: #000000; line-height: 1.25; page-break-after: avoid; }
    h1 { text-align: center; font-size: 18pt; margin-top: 0; }
    h2 { font-size: 14pt; margin-top: 16pt; padding-bottom: 2pt; text-transform: uppercase; }
    h3 { font-size: 12.5pt; margin-top: 12pt; font-weight: 700; }
    p { margin: 6pt 0; }
    ul, ol { margin-top: 5pt; margin-bottom: 7pt; }
    li { margin: 2pt 0; }
    p, li { text-align: justify; }
    code {
      font-family: "Times New Roman", Times, serif;
      background: transparent;
      padding: 0;
      border-radius: 0;
      font-size: 1em;
    }
    pre {
      background: #f8fafc;
      border: 1px solid #e5e7eb;
      border-radius: 6px;
      padding: 7px;
      overflow-x: auto;
      white-space: pre-wrap;
      font-size: 9pt;
      margin: 8px 0;
    }
    table {
      border-collapse: collapse;
      width: 100%;
      margin: 9px 0 13px;
      font-size: 10pt;
    }
    th, td {
      border: 1px solid #d1d5db;
      padding: 4px 6px;
      vertical-align: top;
    }
    th { background: #f2f2f2; font-weight: 700; }
    figure { margin: 10px auto 14px; text-align: center; page-break-inside: avoid; }
    img { max-width: 100%; max-height: 360px; object-fit: contain; }
    figcaption { font-size: 10pt; color: #222222; margin-top: 4px; font-style: italic; }
    hr { border: 0; border-top: 1px solid #999999; margin: 12px 0; }
    """

    html_text = (
        "<!doctype html><html><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
        "<title>YOLO MPI People Count Report</title>"
        f"<style>{css}</style></head><body>{body}</body></html>"
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_text, encoding="utf-8")
    print(f"REPORT_HTML={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
