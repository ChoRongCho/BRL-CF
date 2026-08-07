#!/usr/bin/env python3
"""Write a markdown summary of system log directory file counts."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DOMAINS = ("tomato", "wastesorting")
OUTPUT = ROOT / "check_files_resutl.md"


def count_direct_files(path: Path) -> int:
    return sum(1 for child in path.iterdir() if child.is_file())


def count_total_files(path: Path) -> int:
    return sum(1 for child in path.rglob("*") if child.is_file())


def relative_depth(path: Path, root: Path) -> int:
    return len(path.relative_to(root).parts)


def format_dir_name(path: Path, domain_root: Path) -> str:
    rel = path.relative_to(domain_root)
    if rel.parts == ():
        return "."
    indent = "&nbsp;" * 4 * (len(rel.parts) - 1)
    return f"{indent}`{rel.as_posix()}`"


def build_domain_section(domain: str) -> list[str]:
    domain_root = ROOT / domain
    lines: list[str] = [f"## {domain}", ""]

    if not domain_root.exists():
        lines.extend(["Directory not found.", ""])
        return lines

    all_dirs = [domain_root]
    all_dirs.extend(sorted(path for path in domain_root.rglob("*") if path.is_dir()))

    lines.append(f"- Total files: `{count_total_files(domain_root)}`")
    lines.append(f"- Total directories: `{len(all_dirs)}`")
    lines.append("")
    lines.append("| Directory | Direct files | Total files |")
    lines.append("|---|---:|---:|")

    for path in all_dirs:
        # Keep the report focused on the experiment hierarchy.
        if relative_depth(path, domain_root) > 2:
            continue
        lines.append(
            f"| {format_dir_name(path, domain_root)} "
            f"| {count_direct_files(path)} "
            f"| {count_total_files(path)} |"
        )

    lines.append("")
    return lines


def main() -> None:
    lines = [
        "# System Log File Check",
        "",
        f"- Generated: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`",
        f"- Root: `{ROOT}`",
        "",
    ]

    for domain in DOMAINS:
        lines.extend(build_domain_section(domain))

    OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
