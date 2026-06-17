from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw

from scenario_condition_accuracy import (
    CONDITIONS,
    CONDITION_LABELS,
    DEFAULT_INPUT_DIR,
    DEFAULT_OUTPUT_DIR,
    DOMAIN_LABELS,
    SCENARIOS,
    color_for_accuracy,
    load_font,
    parse_records,
    text_color_for_bg,
    write_long_records,
    write_matrix_csv,
)


def aggregate_task_success(
    records: list[dict[str, str | int | float]],
) -> dict[tuple[str, str, str], dict[str, float]]:
    grouped: dict[tuple[str, str, str], dict[str, float]] = {}
    task_records: list[dict[str, str | int | float]] = []

    for record in records:
        correct = int(record["correct"])
        total = int(record["total"])
        success = int(total > 0 and correct == total)
        enriched = dict(record)
        enriched["task_success"] = success
        task_records.append(enriched)

        keys = [
            (str(record["domain_code"]), str(record["scenario_id"]), str(record["condition_id"])),
            ("Combined", str(record["scenario_id"]), str(record["condition_id"])),
        ]
        for key in keys:
            stats = grouped.setdefault(key, {"correct": 0.0, "total": 0.0})
            stats["correct"] += success
            stats["total"] += 1

    for stats in grouped.values():
        stats["accuracy"] = stats["correct"] / stats["total"] if stats["total"] else 0.0

    return grouped, task_records


def draw_task_success_heatmap(
    output_path: Path,
    grouped: dict[tuple[str, str, str], dict[str, float]],
    domain: str,
) -> None:
    title = f"Scenario-Condition Task Success ({DOMAIN_LABELS[domain]})"
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
        x = margin_left + c_idx * cell_w + cell_w // 2
        draw.text(
            (x, margin_top - 28),
            CONDITION_LABELS[condition],
            font=tick_font,
            fill=(35, 35, 35),
            anchor="mm",
        )

    for r_idx, scenario in enumerate(SCENARIOS):
        y = margin_top + r_idx * cell_h + cell_h // 2
        draw.text(
            (margin_left - 18, y),
            f"S{scenario}",
            font=tick_font,
            fill=(35, 35, 35),
            anchor="rm",
        )

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

    draw.text((legend_x, legend_y - 28), "Task success", font=tick_font, fill=(35, 35, 35), anchor="lm")
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
    parser = argparse.ArgumentParser(
        description="Build 5x5 scenario-condition task-success heatmaps."
    )
    parser.add_argument("--input_dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--missing",
        choices=("wrong", "exclude"),
        default="wrong",
        help="How to handle blank user answers when a correct answer exists.",
    )
    args = parser.parse_args()

    answer_records = parse_records(args.input_dir, args.missing)
    grouped, task_records = aggregate_task_success(answer_records)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_long_records(args.output_dir / "scenario_condition_task_success_records.csv", task_records)

    for domain in ("R", "T", "Combined"):
        prefix = {
            "R": "waste",
            "T": "tomato",
            "Combined": "combined",
        }[domain]
        write_matrix_csv(args.output_dir / f"{prefix}_task_success_5x5.csv", grouped, domain, "accuracy")
        write_matrix_csv(args.output_dir / f"{prefix}_task_success_counts_5x5.csv", grouped, domain, "count")
        draw_task_success_heatmap(args.output_dir / f"{prefix}_task_success_5x5.png", grouped, domain)

    print(f"task records: {len(task_records)}")
    print(f"output_dir: {args.output_dir}")


if __name__ == "__main__":
    main()
