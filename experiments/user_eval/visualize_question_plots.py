from __future__ import annotations

import argparse
import csv
import math
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_PATH = BASE_DIR / "data" / "03_user_survey" / "output.csv"
DEFAULT_OUTPUT_DIR = BASE_DIR / "figure" / "question_plots"

CONDITION_ORDER = ("C1: All", "C2: No", "C3: Ours1", "C4: Ours2", "C5: KnowNo")
CONDITION_LABELS = {
    "C1: All": "C1\nAll",
    "C2: No": "C2\nNo",
    "C3: Ours1": "C3\nOurs1",
    "C4: Ours2": "C4\nOurs2",
    "C5: KnowNo": "C5\nKnowNo",
}
DOMAIN_PANELS = ("토마토 수확", "분리수거", "통합")
DOMAIN_LABELS = {
    "토마토 수확": "Tomato",
    "분리수거": "Waste sorting",
    "통합": "Combined",
}

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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def parse_float(value: str | None) -> float | None:
    if value is None or not value.strip():
        return None
    try:
        return float(value)
    except ValueError:
        return None


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = BOLD_FONT_CANDIDATES if bold else FONT_CANDIDATES
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def item_sort_key(item_id: str) -> tuple[int, int]:
    if item_id.startswith("Q") and item_id[1:].isdigit():
        return (0, int(item_id[1:]))
    if item_id.startswith("N") and item_id[1:].isdigit():
        return (1, int(item_id[1:]))
    if item_id == "F":
        return (2, 0)
    if item_id.startswith("S") and item_id[1:].isdigit():
        return (3, int(item_id[1:]))
    return (4, 999)


def grouped_rows(rows: list[dict[str, str]]) -> dict[tuple[str, str], list[dict[str, str]]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        scale = row.get("척도", "")
        item_id = row.get("문항", "")
        if not scale or not item_id:
            continue
        grouped.setdefault((scale, item_id), []).append(row)
    return grouped


def row_for(rows: list[dict[str, str]], panel: str, condition: str) -> dict[str, str] | None:
    for row in rows:
        if row.get("도메인패널") == panel and row.get("조건 선택") == condition:
            return row
    return None


def draw_wrapped_center(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
    width: int,
    line_gap: int = 5,
) -> int:
    lines = textwrap.wrap(text, width=width, break_long_words=False, replace_whitespace=False) or [text]
    cursor = y
    for line in lines:
        draw.text((x, cursor), line, font=font, fill=fill, anchor="ma")
        bbox = draw.textbbox((x, cursor), line, font=font, anchor="ma")
        cursor += bbox[3] - bbox[1] + line_gap
    return cursor


def draw_vertical_text(
    image: Image.Image,
    text: str,
    center: tuple[int, int],
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
) -> None:
    scratch = Image.new("RGBA", (1, 1))
    bbox = ImageDraw.Draw(scratch).textbbox((0, 0), text, font=font)
    label = Image.new("RGBA", (bbox[2] - bbox[0] + 18, bbox[3] - bbox[1] + 18), (255, 255, 255, 0))
    label_draw = ImageDraw.Draw(label)
    label_draw.text((label.width // 2, label.height // 2), text, font=font, fill=fill, anchor="mm")
    rotated = label.rotate(90, expand=True)
    image.paste(rotated, (center[0] - rotated.width // 2, center[1] - rotated.height // 2), rotated)


def tick_values(y_min: float, y_max: float) -> list[float]:
    if y_max <= 1:
        return [i / 5 for i in range(6)]
    return [float(i) for i in range(int(math.floor(y_min)), int(math.ceil(y_max)) + 1)]


def y_position(value: float, y_min: float, y_max: float, plot_top: int, plot_bottom: int) -> int:
    if y_max == y_min:
        return plot_bottom
    bounded = max(y_min, min(y_max, value))
    return plot_bottom - int((bounded - y_min) / (y_max - y_min) * (plot_bottom - plot_top))


def plot_item(rows: list[dict[str, str]], output_dir: Path) -> Path:
    scale = rows[0]["척도"]
    item_id = rows[0]["문항"]
    item_title = rows[0]["문항내용"]
    y_min = min(parse_float(row.get("값범위최소")) or 0.0 for row in rows)
    y_max = max(parse_float(row.get("값범위최대")) or 1.0 for row in rows)
    if scale == "SAGAT":
        title_prefix = "SAGAT"
        y_label = "Mean correctness"
        file_prefix = "sagat"
        y_min, y_max = 0.0, 1.0
    else:
        title_prefix = "Survey"
        y_label = "Mean response"
        file_prefix = "scale"
        y_min, y_max = 0.0, max(7.0, y_max)

    width, height = 2100, 760
    margin_left, margin_right = 112, 44
    margin_top, margin_bottom = 162, 128
    panel_gap = 54
    panel_width = (width - panel_gap * 2) // 3

    colors = {
        "bar": (54, 126, 184),
        "bar_edge": (31, 77, 117),
        "axis": (35, 35, 35),
        "grid": (222, 226, 232),
        "text": (25, 25, 25),
        "muted": (112, 112, 112),
        "error": (20, 20, 20),
    }
    title_font = load_font(30, bold=True)
    panel_font = load_font(23, bold=True)
    axis_font = load_font(20)
    tick_font = load_font(17)
    value_font = load_font(17, bold=True)

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw_wrapped_center(
        draw,
        width // 2,
        24,
        f"{title_prefix} - {item_title}",
        title_font,
        colors["text"],
        width=76,
    )

    for panel_index, panel in enumerate(DOMAIN_PANELS):
        panel_x0 = panel_index * (panel_width + panel_gap)
        panel_x1 = panel_x0 + panel_width
        plot_left = panel_x0 + margin_left
        plot_right = panel_x1 - margin_right
        plot_top = margin_top
        plot_bottom = height - margin_bottom

        draw.text(
            ((panel_x0 + panel_x1) // 2, 122),
            DOMAIN_LABELS.get(panel, panel),
            font=panel_font,
            fill=colors["text"],
            anchor="mm",
        )

        for tick in tick_values(y_min, y_max):
            y = y_position(tick, y_min, y_max, plot_top, plot_bottom)
            draw.line((plot_left, y, plot_right, y), fill=colors["grid"], width=1)
            tick_label = f"{tick:.1f}" if y_max <= 1 else f"{tick:g}"
            draw.text((plot_left - 12, y), tick_label, font=tick_font, fill=colors["text"], anchor="rm")

        draw.line((plot_left, plot_top, plot_left, plot_bottom), fill=colors["axis"], width=2)
        draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=colors["axis"], width=2)

        usable_left = plot_left + 42
        usable_right = plot_right - 24
        x_points = [
            usable_left + int(i * (usable_right - usable_left) / (len(CONDITION_ORDER) - 1))
            for i in range(len(CONDITION_ORDER))
        ]
        bar_width = max(46, int((usable_right - usable_left) / len(CONDITION_ORDER) * 0.64))

        for x, condition in zip(x_points, CONDITION_ORDER):
            label_lines = CONDITION_LABELS.get(condition, condition).splitlines()
            for line_index, label in enumerate(label_lines):
                draw.text((x, plot_bottom + 26 + line_index * 20), label, font=tick_font, fill=colors["text"], anchor="ma")

            row = row_for(rows, panel, condition)
            value = parse_float(row.get("평균")) if row else None
            if value is None:
                draw.text((x, plot_top + 24), "NA", font=value_font, fill=colors["muted"], anchor="mm")
                continue

            sd = parse_float(row.get("표준편차")) or 0.0
            count = int(float(row.get("표본수", "0")))
            y = y_position(value, y_min, y_max, plot_top, plot_bottom)
            draw.rectangle(
                (x - bar_width // 2, y, x + bar_width // 2, plot_bottom),
                fill=colors["bar"],
                outline=colors["bar_edge"],
                width=2,
            )

            label_y = y - 24
            if count > 1 and sd > 0:
                upper_y = y_position(value + sd, y_min, y_max, plot_top, plot_bottom)
                lower_y = y_position(value - sd, y_min, y_max, plot_top, plot_bottom)
                cap = max(12, bar_width // 4)
                draw.line((x, upper_y, x, lower_y), fill=colors["error"], width=3)
                draw.line((x - cap, upper_y, x + cap, upper_y), fill=colors["error"], width=3)
                draw.line((x - cap, lower_y, x + cap, lower_y), fill=colors["error"], width=3)
                label_y = upper_y - 24
            draw.text((x, max(plot_top + 20, label_y)), f"{value:.2f}", font=value_font, fill=colors["text"], anchor="mm")

        draw.text(((plot_left + plot_right) // 2, height - 30), "Condition", font=axis_font, fill=colors["text"], anchor="mm")
        if panel_index == 0:
            draw_vertical_text(
                image,
                y_label,
                (32, (plot_top + plot_bottom) // 2),
                axis_font,
                colors["text"],
            )

    output_path = output_dir / f"{file_prefix}_{item_id}_condition_mean.png"
    image.save(output_path)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot condition means from output.csv.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_csv(args.input)
    groups = grouped_rows(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for old_plot in args.output_dir.glob("*_condition_mean.png"):
        old_plot.unlink()

    output_paths = [
        plot_item(group, args.output_dir)
        for (_, item_id), group in sorted(groups.items(), key=lambda item: item_sort_key(item[0][1]))
    ]

    sagat_count = sum(1 for (scale, _item_id) in groups if scale == "SAGAT")
    survey_count = sum(1 for (scale, _item_id) in groups if scale == "Survey")
    print(f"input: {args.input}")
    print(f"plots: {args.output_dir}")
    print(f"plot files: {len(output_paths)}")
    print(f"SAGAT plots: {sagat_count}")
    print(f"Survey plots: {survey_count}")


if __name__ == "__main__":
    main()
