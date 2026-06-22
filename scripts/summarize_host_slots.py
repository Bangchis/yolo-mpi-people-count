from __future__ import annotations

import argparse
import re
from pathlib import Path


def parse_hosts(path: Path) -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = []
    if not path.exists():
        return rows
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        host = parts[0]
        slots = 1
        for part in parts[1:]:
            match = re.match(r"slots=(\d+)$", part)
            if match:
                slots = int(match.group(1))
                break
        rows.append((host, slots))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hostfile", default="configs/hosts_macos_gpu")
    parser.add_argument("--output", default="results/evidence/host_slots.csv")
    parser.add_argument("--summary", default="")
    args = parser.parse_args()

    hostfile = Path(args.hostfile)
    output = Path(args.output)
    summary = Path(args.summary) if args.summary else output.with_suffix(".env")
    rows = parse_hosts(hostfile)
    total = sum(slots for _, slots in rows)

    output.parent.mkdir(parents=True, exist_ok=True)
    summary.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        f.write("host,slots\n")
        for host, slots in rows:
            f.write(f"{host},{slots}\n")
    summary.write_text(
        "\n".join([
            f"HOSTFILE={hostfile}",
            f"HOST_COUNT={len(rows)}",
            f"HOSTFILE_SLOT_TOTAL={total}",
            "",
        ]),
        encoding="utf-8",
    )
    print(f"HOST_SLOTS_CSV={output}")
    print(f"HOST_SLOTS_SUMMARY={summary}")
    print(f"HOSTFILE_SLOT_TOTAL={total}")
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())

