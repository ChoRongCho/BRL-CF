from __future__ import annotations

import argparse
import csv
import math
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_PATH = BASE_DIR / "output.csv"
DEFAULT_OUTPUT_DIR = BASE_DIR / "question_plots"

CONDITION_ORDER = ("C1: All", "C2: No", "C3: Ours1", "C4: Ours2", "C5: KnowNo")
CONDITION_LABELS = ("C1=All", "C2=No", "C3=Ours", "C4=Ours2", "C5=KnowNo")
DOMAIN_PANELS = ("토마토 수확", "분리수거", "통합")
DOMAIN_PLOT_LABELS = {
    "토마토 수확": "Tomato",
    "분리수거": "Waste sorting",
    "통합": "Combined",
}
FONT_CANDIDATES = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/roboto/unhinted/RobotoTTF/Roboto-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/opentype/ipaexfont-gothic/ipaexg.ttf",
    "/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf",
)
BOLD_FONT_CANDIDATES = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/roboto/unhinted/RobotoTTF/Roboto-Bold.ttf",
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
    return float(value)


def item_sort_key(item_id: str) -> tuple[int, int]:
    if item_id.startswith("Q") and item_id[1:].isdigit():
        return (0, int(item_id[1:]))
    if item_id.startswith("N") and item_id[1:].isdigit():
        return (1, int(item_id[1:]))
    if item_id == "F":
        return (2, 0)
    if item_id.startswith("S") and item_id[1:].isdigit():
        return (3, int(item_id[1:]))
    return (4, 0)


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = BOLD_FONT_CANDIDATES if bold else FONT_CANDIDATES
    for font_path in candidates:
        if Path(font_path).exists():
            return ImageFont.truetype(font_path, size=size)
    return ImageFont.load_default()


def draw_centered_wrapped_text(
    draw: ImageDraw.ImageDraw,
    center_x: int,
    top_y: int,
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
    max_chars: int,
    line_spacing: int = 6,
) -> int:
    lines = textwrap.wrap(text, width=max_chars, break_long_words=False, replace_whitespace=False) or [text]
    y = top_y
    for line in lines:
        draw.text((center_x, y), line, fill=fill, font=font, anchor="ma")
        bbox = draw.textbbox((center_x, y), line, font=font, anchor="ma")
        y += bbox[3] - bbox[1] + line_spacing
    return y


def draw_vertical_label(
    image: Image.Image,
    center: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
) -> None:
    text_bbox = ImageDraw.Draw(Image.new("RGBA", (1, 1))).textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0] + 12
    text_height = text_bbox[3] - text_bbox[1] + 12
    label_image = Image.new("RGBA", (text_width, text_height), (255, 255, 255, 0))
    label_draw = ImageDraw.Draw(label_image)
    label_draw.text((text_width // 2, text_height // 2), text, fill=fill, font=font, anchor="mm")
    rotated = label_image.rotate(90, expand=True)
    image.paste(
        rotated,
        (center[0] - rotated.width // 2, center[1] - rotated.height // 2),
        rotated,
    )


def group_rows(rows: list[dict[str, str]]) -> dict[tuple[str, str], list[dict[str, str]]]:
    groups: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault((row["척도"], row["문항"]), []).append(row)
    return groups


def row_for(
    rows: list[dict[str, str]],
    panel_label: str,
    condition: str,
) -> dict[str, str] | None:
    for row in rows:
        if row["도메인패널"] == panel_label and row["조건 선택"] == condition:
            return row
    return None


def plot_item(rows: list[dict[str, str]], output_dir: Path) -> Path:
    scale = rows[0]["척도"]
    item_id = rows[0]["문항"]
    item_title = rows[0]["문항내용"]
    y_min = float(rows[0]["값범위최소"])
    y_max = max(float(row["값범위최대"]) for row in rows)
    title_prefix = "SAGAT" if scale == "SAGAT" else "Survey"
    file_prefix = "sagat" if scale == "SAGAT" else "scale"
    y_label = "Mean correctness" if scale == "SAGAT" else "Mean coded response"

    width, height = 2100, 720
    margin_left, margin_right = 110, 42
    margin_top, margin_bottom = 150, 112
    panel_gap = 54
    panel_width = (width - panel_gap * 2) // 3
    bar_color = (54, 126, 184)
    bar_outline_color = (32, 83, 130)
    error_bar_color = (25, 25, 25)
    axis_color = (35, 35, 35)
    grid_color = (218, 224, 231)
    text_color = (25, 25, 25)
    muted_text_color = (110, 110, 110)
    title_font = load_font(30, bold=True)
    panel_font = load_font(23, bold=True)
    axis_font = load_font(20)
    tick_font = load_font(17)
    value_font = load_font(17, bold=True)

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw_centered_wrapped_text(
        draw=draw,
        center_x=width // 2,
        top_y=24,
        text=f"{title_prefix} - {item_title}",
        font=title_font,
        fill=text_color,
        max_chars=74,
    )

    for panel_index, panel_label in enumerate(DOMAIN_PANELS):
        panel_x0 = panel_index * (panel_width + panel_gap)
        panel_x1 = panel_x0 + panel_width
        plot_x0 = panel_x0 + margin_left
        plot_x1 = panel_x1 - margin_right
        plot_y0 = margin_top
        plot_y1 = height - margin_bottom

        draw.text(
            ((panel_x0 + panel_x1) // 2, 112),
            DOMAIN_PLOT_LABELS[panel_label],
            fill=text_color,
            font=panel_font,
            anchor="mm",
        )

        tick_values = [tick / 5 for tick in range(6)] if y_max <= 1.0 else [
            float(tick) for tick in range(int(y_min), int(math.ceil(y_max)) + 1)
        ]
        for value in tick_values:
            y = plot_y1 - int((value - y_min) / (y_max - y_min) * (plot_y1 - plot_y0))
            draw.line((plot_x0, y, plot_x1, y), fill=grid_color, width=1)
            tick_label = f"{value:.1f}" if y_max <= 1.0 else f"{value:g}"
            draw.text((plot_x0 - 12, y), tick_label, fill=text_color, font=tick_font, anchor="rm")

        draw.line((plot_x0, plot_y0, plot_x0, plot_y1), fill=axis_color, width=2)
        draw.line((plot_x0, plot_y1, plot_x1, plot_y1), fill=axis_color, width=2)

        x_inner_left = plot_x0 + 42
        x_inner_right = plot_x1 - 24
        x_points = [
            x_inner_left + int(index * (x_inner_right - x_inner_left) / (len(CONDITION_ORDER) - 1))
            for index in range(len(CONDITION_ORDER))
        ]
        bar_width = max(46, int((x_inner_right - x_inner_left) / len(CONDITION_ORDER) * 0.70))

        for x, label, condition in zip(x_points, CONDITION_LABELS, CONDITION_ORDER):
            draw.text((x, plot_y1 + 28), label, fill=text_color, font=tick_font, anchor="ma")
            row = row_for(rows, panel_label, condition)
            value = parse_float(row["평균"]) if row is not None else None
            if value is None:
                draw.text((x, plot_y0 + 24), "NA", fill=muted_text_color, font=value_font, anchor="mm")
                continue

            sd = parse_float(row["표준편차"]) or 0.0
            count = int(row["표본수"])
            y = plot_y1 - int((max(y_min, min(y_max, value)) - y_min) / (y_max - y_min) * (plot_y1 - plot_y0))
            draw.rectangle(
                (x - bar_width // 2, y, x + bar_width // 2, plot_y1),
                fill=bar_color,
                outline=bar_outline_color,
                width=2,
            )
            if count > 1 and sd > 0:
                upper_y = plot_y1 - int((min(y_max, value + sd) - y_min) / (y_max - y_min) * (plot_y1 - plot_y0))
                lower_y = plot_y1 - int((max(y_min, value - sd) - y_min) / (y_max - y_min) * (plot_y1 - plot_y0))
                cap_half_width = max(12, bar_width // 4)
                draw.line((x, upper_y, x, lower_y), fill=error_bar_color, width=3)
                draw.line((x - cap_half_width, upper_y, x + cap_half_width, upper_y), fill=error_bar_color, width=3)
                draw.line((x - cap_half_width, lower_y, x + cap_half_width, lower_y), fill=error_bar_color, width=3)
                label_y = max(plot_y0 + 20, upper_y - 24)
            else:
                label_y = max(plot_y0 + 20, y - 24)
            draw.text((x, label_y), f"{value:.2f}", fill=text_color, font=value_font, anchor="mm")

        draw.text(
            ((plot_x0 + plot_x1) // 2, height - 28),
            "Condition",
            fill=text_color,
            font=axis_font,
            anchor="mm",
        )
        if panel_index == 0:
            draw_vertical_label(
                image=image,
                center=(30, (plot_y0 + plot_y1) // 2),
                text=y_label,
                font=axis_font,
                fill=text_color,
            )

    output_path = output_dir / f"{file_prefix}_{item_id}_condition_mean.png"
    image.save(output_path)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot question means from output.csv.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_csv(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for old_plot in args.output_dir.glob("*_condition_mean.png"):
        old_plot.unlink()

    output_paths = [
        plot_item(group, args.output_dir)
        for _, group in sorted(group_rows(rows).items(), key=lambda item: item_sort_key(item[0][1]))
    ]

    print(f"input: {args.input}")
    print(f"plots: {args.output_dir}")
    print(f"plot files: {len(output_paths)}")


if __name__ == "__main__":
    main()
