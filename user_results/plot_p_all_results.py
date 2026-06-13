from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DEFAULT_INPUT_PATH = PROJECT_ROOT / "users_raw_data" / "피험자별결과.csv"
DEFAULT_OUTPUT_DIR = BASE_DIR / "figures"

CONDITIONS = ("C1: All", "C2: No", "C3: Ours1", "C4: Ours2", "C5: KnowNo")
CONDITION_LABELS = {
    "C1: All": "C1\nAll",
    "C2: No": "C2\nNo",
    "C3: Ours1": "C3\nOurs1",
    "C4: Ours2": "C4\nOurs2",
    "C5: KnowNo": "C5\nKnowNo",
}
DOMAINS = ("Waste", "Tomato")
METRICS = (
    "조작성공률 (%)",
    "조작시간 (s)",
    "SAGAT1",
    "SAGAT2",
    "SAGAT3",
    "Fatigue",
    "NASA-RTLX",
    "SART",
)
SAGAT_METRICS = ("SAGAT1", "SAGAT2", "SAGAT3")
METRIC_FILE_STEMS = {
    "조작성공률 (%)": "p_all_task_success",
    "조작시간 (s)": "p_all_operation_time",
    "SAGAT1": "p_all_sagat1",
    "SAGAT2": "p_all_sagat2",
    "SAGAT3": "p_all_sagat3",
    "Fatigue": "p_all_fatigue",
    "NASA-RTLX": "p_all_nasa_rtlx",
    "SART": "p_all_sart",
}
Y_LABELS = {
    "조작성공률 (%)": "Task success (%)",
    "조작시간 (s)": "Operation time (s)",
    "SAGAT1": "Mean correctness",
    "SAGAT2": "Mean correctness",
    "SAGAT3": "Mean correctness",
    "Fatigue": "Score (lower is better)",
    "NASA-RTLX": "Score (lower is better)",
    "SART": "Score (lower is better)",
}
FIXED_Y_LIMITS = {
    "조작성공률 (%)": (0.0, 100.0),
    "SAGAT1": (0.0, 1.0),
    "SAGAT2": (0.0, 1.0),
    "SAGAT3": (0.0, 1.0),
    "Fatigue": (0.0, 7.0),
    "NASA-RTLX": (0.0, 7.0),
    "SART": (0.0, 7.0),
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


def normalize(value: str | None) -> str:
    return " ".join((value or "").strip().split())


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = BOLD_FONT_CANDIDATES if bold else FONT_CANDIDATES
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def parse_float(value: str | None) -> float | None:
    text = normalize(value)
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def read_rows(path: Path) -> list[list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.reader(file))


def load_p_all(path: Path) -> dict[tuple[str, str, str], float]:
    rows = read_rows(path)
    start_index = None
    for index, row in enumerate(rows):
        if len(row) >= 3 and normalize(row[2]) == "P_ALL":
            start_index = index
            break
    if start_index is None:
        raise ValueError(f"P_ALL block was not found: {path}")
    if start_index + 1 >= len(rows):
        raise ValueError(f"P_ALL metric header row is missing: {path}")

    metrics = [normalize(metric) for metric in rows[start_index + 1][2:]]
    result: dict[tuple[str, str, str], float] = {}
    current_condition = ""
    for row in rows[start_index + 2:]:
        if not any(normalize(cell) for cell in row):
            break
        if normalize(row[0]):
            current_condition = normalize(row[0])
        domain = normalize(row[1] if len(row) > 1 else "")
        if current_condition not in CONDITIONS or domain not in DOMAINS:
            continue
        for metric_index, metric in enumerate(metrics, start=2):
            if metric not in METRICS:
                continue
            value = parse_float(row[metric_index] if metric_index < len(row) else "")
            if value is not None:
                result[(metric, domain, current_condition)] = value
    return result


def y_limits(metric: str, values: list[float]) -> tuple[float, float]:
    if metric in FIXED_Y_LIMITS:
        return FIXED_Y_LIMITS[metric]
    if not values:
        return 0.0, 1.0
    top = max(values)
    if top <= 0:
        return 0.0, 1.0
    return 0.0, math.ceil(top * 1.15 / 5) * 5


def tick_values(y_min: float, y_max: float) -> list[float]:
    if y_max <= 1.0:
        return [index / 5 for index in range(6)]
    if y_max <= 7.0:
        return [float(index) for index in range(0, 8)]
    step = 20.0 if y_max <= 100 else max(5.0, math.ceil(y_max / 5 / 5) * 5)
    tick = y_min
    ticks = []
    while tick <= y_max + 1e-9:
        ticks.append(tick)
        tick += step
    return ticks


def y_pos(value: float, y_min: float, y_max: float, top: int, bottom: int) -> int:
    if y_max == y_min:
        return bottom
    return bottom - int((value - y_min) / (y_max - y_min) * (bottom - top))


def format_number(value: float, metric: str) -> str:
    if metric == "조작성공률 (%)":
        return f"{value:.1f}".rstrip("0").rstrip(".")
    if metric == "조작시간 (s)":
        return f"{value:.1f}".rstrip("0").rstrip(".")
    return f"{value:.2f}".rstrip("0").rstrip(".")


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


def draw_domain_metric(
    metric: str,
    domain: str,
    data: dict[tuple[str, str, str], float],
    output_dir: Path,
) -> Path:
    width, height = 980, 720
    plot_left, plot_right = 120, 930
    plot_top, plot_bottom = 105, 575
    group_width = (plot_right - plot_left) / len(CONDITIONS)
    bar_width = 72
    colors = {
        "Waste": (70, 130, 180),
        "Tomato": (210, 104, 76),
        "axis": (35, 35, 35),
        "grid": (224, 228, 234),
        "text": (24, 24, 24),
        "muted": (116, 116, 116),
    }
    title_font = load_font(30, bold=True)
    axis_font = load_font(21)
    tick_font = load_font(18)
    value_font = load_font(17, bold=True)

    values = [
        value
        for condition in CONDITIONS
        if (value := data.get((metric, domain, condition))) is not None
    ]
    y_min, y_max = y_limits(metric, values)

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((width // 2, 44), f"P_ALL - {metric} ({domain})", font=title_font, fill=colors["text"], anchor="mm")

    for tick in tick_values(y_min, y_max):
        y = y_pos(tick, y_min, y_max, plot_top, plot_bottom)
        draw.line((plot_left, y, plot_right, y), fill=colors["grid"], width=1)
        label = f"{tick:.1f}" if y_max <= 1.0 else f"{tick:g}"
        draw.text((plot_left - 14, y), label, font=tick_font, fill=colors["text"], anchor="rm")

    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill=colors["axis"], width=2)
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=colors["axis"], width=2)
    draw_vertical_text(image, Y_LABELS[metric], (40, (plot_top + plot_bottom) // 2), axis_font, colors["text"])

    for condition_index, condition in enumerate(CONDITIONS):
        center_x = plot_left + group_width * condition_index + group_width / 2
        value = data.get((metric, domain, condition))
        if value is not None:
            x0 = int(center_x - bar_width / 2)
            x1 = x0 + bar_width
            y0 = y_pos(value, y_min, y_max, plot_top, plot_bottom)
            draw.rectangle((x0, y0, x1, plot_bottom), fill=colors[domain], outline=(45, 45, 45), width=1)
            draw.text(
                ((x0 + x1) // 2, y0 - 12),
                format_number(value, metric),
                font=value_font,
                fill=colors["text"],
                anchor="mm",
            )
        for line_index, line in enumerate(CONDITION_LABELS[condition].splitlines()):
            draw.text(
                (center_x, plot_bottom + 35 + line_index * 22),
                line,
                font=tick_font,
                fill=colors["text"],
                anchor="mm",
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{METRIC_FILE_STEMS[metric]}_{domain.lower()}.png"
    image.save(output_path)
    return output_path


def draw_sagat_grid(data: dict[tuple[str, str, str], float], output_dir: Path) -> Path:
    width, height = 1320, 1380
    margin_left, margin_right = 95, 55
    margin_top, margin_bottom = 125, 80
    col_gap, row_gap = 70, 82
    panel_width = (width - margin_left - margin_right - col_gap) // 2
    panel_height = (height - margin_top - margin_bottom - row_gap * 2) // 3
    colors = {
        "Waste": (70, 130, 180),
        "Tomato": (210, 104, 76),
        "axis": (35, 35, 35),
        "grid": (224, 228, 234),
        "text": (24, 24, 24),
    }
    title_font = load_font(32, bold=True)
    panel_font = load_font(23, bold=True)
    tick_font = load_font(16)
    value_font = load_font(15, bold=True)

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((width // 2, 44), "P_ALL - SAGAT", font=title_font, fill=colors["text"], anchor="mm")

    for row_index, metric in enumerate(SAGAT_METRICS):
        for col_index, domain in enumerate(DOMAINS):
            x0 = margin_left + col_index * (panel_width + col_gap)
            y0 = margin_top + row_index * (panel_height + row_gap)
            plot_left = x0 + 70
            plot_right = x0 + panel_width - 20
            plot_top = y0 + 42
            plot_bottom = y0 + panel_height - 72
            group_width = (plot_right - plot_left) / len(CONDITIONS)
            bar_width = 40

            title = f"{metric} - {domain}"
            draw.text((x0 + panel_width // 2, y0 + 15), title, font=panel_font, fill=colors["text"], anchor="mm")

            for tick in tick_values(0.0, 1.0):
                y = y_pos(tick, 0.0, 1.0, plot_top, plot_bottom)
                draw.line((plot_left, y, plot_right, y), fill=colors["grid"], width=1)
                draw.text((plot_left - 10, y), f"{tick:.1f}", font=tick_font, fill=colors["text"], anchor="rm")

            draw.line((plot_left, plot_top, plot_left, plot_bottom), fill=colors["axis"], width=2)
            draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=colors["axis"], width=2)

            for condition_index, condition in enumerate(CONDITIONS):
                value = data.get((metric, domain, condition))
                center_x = plot_left + group_width * condition_index + group_width / 2
                if value is not None:
                    bar_x0 = int(center_x - bar_width / 2)
                    bar_x1 = bar_x0 + bar_width
                    bar_y0 = y_pos(value, 0.0, 1.0, plot_top, plot_bottom)
                    draw.rectangle((bar_x0, bar_y0, bar_x1, plot_bottom), fill=colors[domain], outline=(45, 45, 45), width=1)
                    draw.text(
                        ((bar_x0 + bar_x1) // 2, bar_y0 - 10),
                        format_number(value, metric),
                        font=value_font,
                        fill=colors["text"],
                        anchor="mm",
                    )
                label = CONDITION_LABELS[condition].splitlines()[0]
                draw.text((center_x, plot_bottom + 23), label, font=tick_font, fill=colors["text"], anchor="mm")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "p_all_sagat_2x3.png"
    image.save(output_path)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot P_ALL condition-wise user-study results.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_p_all(args.input)
    output_paths = [
        draw_domain_metric(metric, domain, data, args.output_dir)
        for metric in METRICS
        for domain in DOMAINS
    ]
    output_paths.append(draw_sagat_grid(data, args.output_dir))

    print(f"input: {args.input}")
    print(f"output_dir: {args.output_dir}")
    print(f"figures: {len(output_paths)}")
    for path in output_paths:
        print(path)


if __name__ == "__main__":
    main()
