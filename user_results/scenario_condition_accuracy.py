from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "users_raw_data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "user_results" / "01_op_sr"

HEADER_RE = re.compile(r"^([RT])([1-5])\s*\(\s*s([1-5])\s*\)")
ANSWER_RE = re.compile(r"\b([ABTF])\b", re.IGNORECASE)

DOMAIN_LABELS = {
    "R": "Waste sorting",
    "T": "Tomato harvesting",
    "Combined": "Combined",
}
CONDITION_LABELS = {
    "1": "All",
    "2": "No",
    "3": "Ours1",
    "4": "Ours2",
    "5": "KnowNo",
}
SCENARIOS = ["1", "2", "3", "4", "5"]
CONDITIONS = ["1", "2", "3", "4", "5"]

FONT_CANDIDATES = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/opentype/ipaexfont-gothic/ipaexg.ttf",
    "/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf",
)
BOLD_FONT_CANDIDATES = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/opentype/ipaexfont-gothic/ipaexg.ttf",
    "/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf",
)


@dataclass
class Block:
    source_file: str
    header_row: int
    col: int
    domain: str
    condition: str
    scenario: str
    label: str


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = BOLD_FONT_CANDIDATES if bold else FONT_CANDIDATES
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def read_rows(path: Path) -> list[list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.reader(file))


def cell(row: list[str], idx: int) -> str:
    if 0 <= idx < len(row):
        return row[idx].strip()
    return ""


def is_valid_header(rows: list[list[str]], r: int, c: int) -> bool:
    return cell(rows[r], c + 1) == "정답" and cell(rows[r], c + 2) == "사용자답"


def find_blocks(path: Path, rows: list[list[str]]) -> list[Block]:
    blocks: list[Block] = []
    for r, row in enumerate(rows):
        for c, raw in enumerate(row):
            text = raw.strip()
            match = HEADER_RE.match(text)
            if not match or not is_valid_header(rows, r, c):
                continue
            domain, condition, scenario = match.groups()
            blocks.append(
                Block(
                    source_file=path.name,
                    header_row=r,
                    col=c,
                    domain=domain,
                    condition=condition,
                    scenario=scenario,
                    label=text,
                )
            )
    return blocks


def split_answer_tokens(value: str) -> list[str]:
    if not value or not value.strip():
        return []
    tokens: list[str] = []
    for line in value.replace("\r", "\n").split("\n"):
        stripped = line.strip().upper()
        if not stripped:
            continue
        found = ANSWER_RE.findall(stripped)
        if found:
            tokens.extend(token.upper() for token in found)
    return tokens


def score_answers(correct_cell: str, user_cell: str, missing: str) -> tuple[int, int]:
    correct_tokens = split_answer_tokens(correct_cell)
    user_tokens = split_answer_tokens(user_cell)
    if not correct_tokens:
        return 0, 0

    correct_count = 0
    total = 0
    for idx, expected in enumerate(correct_tokens):
        actual = user_tokens[idx] if idx < len(user_tokens) else None
        if actual is None and missing == "exclude":
            continue
        total += 1
        if actual == expected:
            correct_count += 1
    return correct_count, total


def next_header_row(block: Block, all_header_rows: list[int], row_count: int) -> int:
    for header_row in all_header_rows:
        if header_row > block.header_row:
            return header_row
    return row_count


def parse_records(input_dir: Path, missing: str) -> list[dict[str, str | int | float]]:
    records: list[dict[str, str | int | float]] = []
    paths = sorted(input_dir.glob("BRL_논문_타임라인 정리.xlsx - 6월*일.csv"))
    if not paths:
        raise FileNotFoundError(f"No timeline CSV files found under {input_dir}")

    for path in paths:
        rows = read_rows(path)
        blocks = find_blocks(path, rows)
        header_rows = sorted({block.header_row for block in blocks})

        for block in blocks:
            end_row = next_header_row(block, header_rows, len(rows))
            block_correct = 0
            block_total = 0
            question_rows = 0
            for r in range(block.header_row + 1, end_row):
                correct_cell = cell(rows[r], block.col + 1)
                user_cell = cell(rows[r], block.col + 2)
                correct_count, total = score_answers(correct_cell, user_cell, missing)
                if total == 0:
                    continue
                question_rows += 1
                block_correct += correct_count
                block_total += total

            if block_total == 0:
                continue

            records.append(
                {
                    "source_file": block.source_file,
                    "header_row": block.header_row + 1,
                    "column": block.col + 1,
                    "domain_code": block.domain,
                    "domain": DOMAIN_LABELS[block.domain],
                    "condition_id": block.condition,
                    "condition": CONDITION_LABELS[block.condition],
                    "scenario_id": block.scenario,
                    "scenario": f"Scenario {block.scenario}",
                    "label": block.label,
                    "question_rows": question_rows,
                    "correct": block_correct,
                    "total": block_total,
                    "accuracy": block_correct / block_total,
                }
            )
    return records


def aggregate(records: list[dict[str, str | int | float]]) -> dict[tuple[str, str, str], dict[str, float]]:
    grouped: dict[tuple[str, str, str], dict[str, float]] = {}
    for record in records:
        keys = [
            (str(record["domain_code"]), str(record["scenario_id"]), str(record["condition_id"])),
            ("Combined", str(record["scenario_id"]), str(record["condition_id"])),
        ]
        for key in keys:
            stats = grouped.setdefault(key, {"correct": 0.0, "total": 0.0})
            stats["correct"] += float(record["correct"])
            stats["total"] += float(record["total"])

    for stats in grouped.values():
        stats["accuracy"] = stats["correct"] / stats["total"] if stats["total"] else 0.0
    return grouped


def write_long_records(path: Path, records: list[dict[str, str | int | float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(records[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def write_matrix_csv(
    path: Path,
    grouped: dict[tuple[str, str, str], dict[str, float]],
    domain: str,
    metric: str,
) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["scenario"] + [CONDITION_LABELS[c] for c in CONDITIONS])
        for scenario in SCENARIOS:
            row = [f"Scenario {scenario}"]
            for condition in CONDITIONS:
                stats = grouped.get((domain, scenario, condition), {})
                if metric == "accuracy":
                    row.append(f"{stats.get('accuracy', 0.0):.4f}" if stats else "")
                elif metric == "count":
                    row.append(
                        f"{int(stats.get('correct', 0))}/{int(stats.get('total', 0))}"
                        if stats
                        else ""
                    )
            writer.writerow(row)


def color_for_accuracy(value: float | None) -> tuple[int, int, int]:
    if value is None:
        return (236, 238, 242)
    # Color-blind friendly light yellow -> purple scale.
    low = (255, 246, 181)
    high = (150, 126, 184)
    value = max(0.0, min(1.0, value))
    return tuple(int(low[i] + (high[i] - low[i]) * value) for i in range(3))


def text_color_for_bg(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    luminance = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    return (255, 255, 255) if luminance < 150 else (30, 30, 30)


def draw_heatmap(
    output_path: Path,
    grouped: dict[tuple[str, str, str], dict[str, float]],
    domain: str,
) -> None:
    title = f"Scenario-Condition Accuracy ({DOMAIN_LABELS[domain]})"
    width, height = 1180, 920
    margin_left, margin_top = 190, 180
    cell_w, cell_h = 165, 115
    legend_x = margin_left + cell_w * 5 + 45
    legend_y = margin_top

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = load_font(36, bold=True)
    axis_font = load_font(24, bold=True)
    tick_font = load_font(22)
    value_font = load_font(24, bold=True)
    small_font = load_font(18)

    draw.text((width // 2, 48), title, font=title_font, fill=(25, 25, 25), anchor="mm")
    draw.text(
        (margin_left + cell_w * 2.5, 108),
        "Condition",
        font=axis_font,
        fill=(30, 30, 30),
        anchor="mm",
    )
    draw.text(
        (58, margin_top + cell_h * 2.5),
        "Scenario",
        font=axis_font,
        fill=(30, 30, 30),
        anchor="mm",
    )

    for c_idx, condition in enumerate(CONDITIONS):
        label = CONDITION_LABELS[condition]
        x = margin_left + c_idx * cell_w + cell_w // 2
        draw.text((x, margin_top - 28), label, font=tick_font, fill=(35, 35, 35), anchor="mm")

    for r_idx, scenario in enumerate(SCENARIOS):
        y = margin_top + r_idx * cell_h + cell_h // 2
        draw.text((margin_left - 18, y), f"S{scenario}", font=tick_font, fill=(35, 35, 35), anchor="rm")

    for r_idx, scenario in enumerate(SCENARIOS):
        for c_idx, condition in enumerate(CONDITIONS):
            stats = grouped.get((domain, scenario, condition))
            value = None if not stats else float(stats["accuracy"])
            x0 = margin_left + c_idx * cell_w
            y0 = margin_top + r_idx * cell_h
            x1 = x0 + cell_w
            y1 = y0 + cell_h
            bg = color_for_accuracy(value)
            draw.rectangle((x0, y0, x1, y1), fill=bg, outline=(255, 255, 255), width=3)
            if value is None:
                label = "-"
                sublabel = ""
            else:
                label = f"{value * 100:.1f}%"
                sublabel = f"{int(stats['correct'])}/{int(stats['total'])}"
            fill = text_color_for_bg(bg)
            draw.text(((x0 + x1) // 2, y0 + 47), label, font=value_font, fill=fill, anchor="mm")
            if sublabel:
                draw.text(((x0 + x1) // 2, y0 + 78), sublabel, font=small_font, fill=fill, anchor="mm")

    # Legend
    draw.text((legend_x, legend_y - 28), "Accuracy", font=tick_font, fill=(35, 35, 35), anchor="lm")
    legend_h = cell_h * 5
    for i in range(legend_h):
        value = 1.0 - i / max(legend_h - 1, 1)
        draw.line(
            (legend_x, legend_y + i, legend_x + 34, legend_y + i),
            fill=color_for_accuracy(value),
            width=1,
        )
    draw.rectangle((legend_x, legend_y, legend_x + 34, legend_y + legend_h), outline=(100, 100, 100), width=1)
    for value in (1.0, 0.75, 0.5, 0.25, 0.0):
        y = legend_y + int((1.0 - value) * legend_h)
        draw.line((legend_x + 36, y, legend_x + 43, y), fill=(60, 60, 60), width=1)
        draw.text((legend_x + 50, y), f"{value:.2f}", font=small_font, fill=(35, 35, 35), anchor="lm")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    image.save(output_path.with_suffix(".pdf"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build 5x5 scenario-condition accuracy heatmaps.")
    parser.add_argument("--input_dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--missing",
        choices=("wrong", "exclude"),
        default="wrong",
        help="How to handle blank user answers when a correct answer exists.",
    )
    args = parser.parse_args()

    records = parse_records(args.input_dir, args.missing)
    grouped = aggregate(records)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_long_records(args.output_dir / "scenario_condition_answer_records.csv", records)
    for domain in ("R", "T", "Combined"):
        prefix = {
            "R": "waste",
            "T": "tomato",
            "Combined": "combined",
        }[domain]
        write_matrix_csv(args.output_dir / f"{prefix}_accuracy_5x5.csv", grouped, domain, "accuracy")
        write_matrix_csv(args.output_dir / f"{prefix}_counts_5x5.csv", grouped, domain, "count")
        draw_heatmap(args.output_dir / f"{prefix}_accuracy_5x5.png", grouped, domain)

    print(f"records: {len(records)}")
    print(f"output_dir: {args.output_dir}")


if __name__ == "__main__":
    main()
