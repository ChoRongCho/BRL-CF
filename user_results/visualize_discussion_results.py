from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_STATS_PATH = BASE_DIR / "stats_results.csv"
DEFAULT_SCORES_PATH = BASE_DIR / "participant_condition_scores.csv"
DEFAULT_OUTPUT_DIR = BASE_DIR / "discussion_figures"
DEFAULT_TABLE_PATH = BASE_DIR / "discussion_source_table.csv"

CONDITION_ORDER = ("C1: All", "C2: No", "C3: Ours1", "C4: Ours2", "C5: KnowNo")
CONDITION_SHORT = {
    "C1: All": "C1",
    "C2: No": "C2",
    "C3: Ours1": "C3",
    "C4: Ours2": "C4",
    "C5: KnowNo": "C5",
}
CONDITION_LABELS = {
    "C1: All": "All",
    "C2: No": "No",
    "C3: Ours1": "Ours1",
    "C4: Ours2": "Ours2",
    "C5: KnowNo": "KnowNo",
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

KEY_RESULTS = (
    {
        "figure": "fig1_sagat_accuracy_recycling.png",
        "measure": "SAGAT_accuracy",
        "domain_panel": "분리수거",
        "title": "SAGAT accuracy - Recycling",
        "ylabel": "Mean correctness",
        "ylim": (0.0, 1.0),
        "comparisons": ("C3-C1", "C4-C2", "C4-C3"),
        "discussion_point": "Ours1 showed higher selected awareness than All; Ours2 trended lower than No/Ours1.",
    },
    {
        "figure": "fig2_sagat_q4_recycling.png",
        "measure": "SAGAT_Q4",
        "domain_panel": "분리수거",
        "title": "SAGAT Q4 - Recycling",
        "ylabel": "Correctness",
        "ylim": (0.0, 1.0),
        "comparisons": ("C3-C2", "C4-C3"),
        "discussion_point": "Q4 is the strongest source of Ours1's selected awareness signal.",
    },
    {
        "figure": "fig3_sagat_q4_combined.png",
        "measure": "SAGAT_Q4",
        "domain_panel": "통합",
        "title": "SAGAT Q4 - Combined domains",
        "ylabel": "Correctness",
        "ylim": (0.0, 1.0),
        "comparisons": ("C3-C2", "C4-C3"),
        "discussion_point": "The Q4 pattern remains visible after combining domains.",
    },
    {
        "figure": "fig4_workload_combined.png",
        "measure": "NASA_TLX_workload",
        "domain_panel": "통합",
        "title": "NASA-TLX workload - Combined domains",
        "ylabel": "Workload (lower is better)",
        "ylim": (0.0, 4.0),
        "comparisons": ("C3-C2", "C4-C3", "C4-C5"),
        "discussion_point": "Ours2 reduced workload relative to KnowNo and trended lower than Ours1.",
    },
    {
        "figure": "fig5_effort_combined.png",
        "measure": "NASA_N5_effort",
        "domain_panel": "통합",
        "title": "NASA-TLX effort - Combined domains",
        "ylabel": "Effort (lower is better)",
        "ylim": (0.0, 5.0),
        "comparisons": ("C4-C1", "C4-C3", "C4-C5"),
        "discussion_point": "Ours2 reduced perceived effort relative to All and KnowNo.",
    },
    {
        "figure": "fig6_workload_effort_recycling.png",
        "measure": "NASA_N5_effort",
        "domain_panel": "분리수거",
        "title": "NASA-TLX effort - Recycling",
        "ylabel": "Effort (lower is better)",
        "ylim": (0.0, 5.0),
        "comparisons": ("C3-C2", "C4-C5"),
        "discussion_point": "Ours2 reduced recycling effort relative to KnowNo.",
    },
)

TABLE_FIELDS = [
    "discussion_point",
    "measure",
    "domain_panel",
    "test",
    "comparison",
    "n",
    "C1_mean",
    "C1_sd",
    "C2_mean",
    "C2_sd",
    "C3_mean",
    "C3_sd",
    "C4_mean",
    "C4_sd",
    "C5_mean",
    "C5_sd",
    "statistic",
    "df",
    "p",
    "p_holm",
    "effect_size",
    "mean_difference",
    "median_difference",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = BOLD_FONT_CANDIDATES if bold else FONT_CANDIDATES
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def parse_float(value: str | None) -> float:
    if value is None or not value.strip():
        return math.nan
    return float(value)


def format_p(value: str) -> str:
    p = parse_float(value)
    if math.isnan(p):
        return ""
    if p < 0.001:
        return "p<.001"
    return f"p={p:.3f}".replace("0.", ".")


def format_value(value: float, y_max: float) -> str:
    if y_max <= 1.0:
        return f"{value:.2f}"
    return f"{value:.2f}".rstrip("0").rstrip(".")


def stats_key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return (row["measure"], row["domain_panel"], row["test"], row["comparison"])


def build_source_table(stats_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    lookup = {stats_key(row): row for row in stats_rows if row.get("test_family") == "planned"}
    table_rows: list[dict[str, str]] = []
    for spec in KEY_RESULTS:
        measure = spec["measure"]
        panel = spec["domain_panel"]
        point = spec["discussion_point"]
        keys = [("Friedman", "C1-C5")]
        keys.extend(("Wilcoxon signed-rank exact", comparison) for comparison in spec["comparisons"])
        for test, comparison in keys:
            row = lookup.get((measure, panel, test, comparison))
            if row is None:
                continue
            table_rows.append(
                {
                    "discussion_point": str(point),
                    **{field: row.get(field, "") for field in TABLE_FIELDS if field != "discussion_point"},
                }
            )
    return table_rows


def score_lookup(score_rows: list[dict[str, str]]) -> dict[tuple[str, str, str], list[tuple[str, float]]]:
    grouped: dict[tuple[str, str, str], list[tuple[str, float]]] = {}
    for row in score_rows:
        key = (row["measure"], row["domain_panel"], row["condition"])
        grouped.setdefault(key, []).append((row["subject"], parse_float(row["value"])))
    for values in grouped.values():
        values.sort(key=lambda item: item[0])
    return grouped


def plot_spec(
    spec: dict[str, object],
    stats_rows: list[dict[str, str]],
    scores: dict[tuple[str, str, str], list[tuple[str, float]]],
    output_dir: Path,
) -> Path:
    measure = str(spec["measure"])
    panel = str(spec["domain_panel"])
    y_min, y_max = spec["ylim"]  # type: ignore[misc]
    y_min = float(y_min)
    y_max = float(y_max)

    friedman = next(
        row
        for row in stats_rows
        if row["measure"] == measure
        and row["domain_panel"] == panel
        and row["test_family"] == "planned"
        and row["test"] == "Friedman"
    )
    pair_rows = {
        row["comparison"]: row
        for row in stats_rows
        if row["measure"] == measure
        and row["domain_panel"] == panel
        and row["test_family"] == "planned"
        and row["test"] == "Wilcoxon signed-rank exact"
    }

    means = [parse_float(friedman[f"{CONDITION_SHORT[condition]}_mean"]) for condition in CONDITION_ORDER]
    sds = [parse_float(friedman[f"{CONDITION_SHORT[condition]}_sd"]) for condition in CONDITION_ORDER]

    width, height = 1500, 940
    plot_left, plot_right = 150, 1390
    plot_top, plot_bottom = 190, 700
    bar_width = 118
    group_gap = (plot_right - plot_left - bar_width * len(CONDITION_ORDER)) / (len(CONDITION_ORDER) - 1)
    colors = {
        "C1: All": (125, 125, 125),
        "C2: No": (80, 150, 210),
        "C3: Ours1": (45, 155, 95),
        "C4: Ours2": (220, 130, 45),
        "C5: KnowNo": (145, 95, 180),
        "axis": (35, 35, 35),
        "grid": (224, 228, 234),
        "text": (25, 25, 25),
        "muted": (95, 95, 95),
        "participant": (40, 40, 40, 45),
    }
    title_font = load_font(34, bold=True)
    subtitle_font = load_font(23)
    axis_font = load_font(23)
    tick_font = load_font(20)
    small_font = load_font(18)
    bold_small = load_font(18, bold=True)

    def y_pos(value: float) -> int:
        return plot_bottom - int((value - y_min) / (y_max - y_min) * (plot_bottom - plot_top))

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image, "RGBA")

    draw.text((width // 2, 38), str(spec["title"]), font=title_font, fill=colors["text"], anchor="ma")
    subtitle = f"Friedman chi2(4)={parse_float(friedman['statistic']):.2f}, {format_p(friedman['p'])}, W={parse_float(friedman['effect_size']):.2f}"
    draw.text((width // 2, 88), subtitle, font=subtitle_font, fill=colors["muted"], anchor="ma")

    tick_count = 5 if y_max <= 1 else int(y_max - y_min)
    for index in range(tick_count + 1):
        tick = y_min + (y_max - y_min) * index / tick_count
        y = y_pos(tick)
        draw.line((plot_left, y, plot_right, y), fill=colors["grid"], width=1)
        draw.text((plot_left - 18, y), format_value(tick, y_max), font=tick_font, fill=colors["text"], anchor="rm")

    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill=colors["axis"], width=2)
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=colors["axis"], width=2)
    draw.text((plot_left, plot_top - 34), str(spec["ylabel"]), font=axis_font, fill=colors["text"], anchor="la")

    centers: dict[str, int] = {}
    for index, condition in enumerate(CONDITION_ORDER):
        x0 = int(plot_left + index * (bar_width + group_gap))
        x1 = x0 + bar_width
        center = (x0 + x1) // 2
        centers[condition] = center
        mean_value = means[index]
        sd_value = sds[index]
        y_mean = y_pos(mean_value)
        y_zero = y_pos(max(y_min, 0.0))
        draw.rectangle((x0, y_mean, x1, y_zero), fill=colors[condition] + (220,), outline=colors[condition], width=2)

        y_error_top = y_pos(min(y_max, mean_value + sd_value))
        y_error_bottom = y_pos(max(y_min, mean_value - sd_value))
        draw.line((center, y_error_top, center, y_error_bottom), fill=(25, 25, 25), width=3)
        draw.line((center - 20, y_error_top, center + 20, y_error_top), fill=(25, 25, 25), width=3)
        draw.line((center - 20, y_error_bottom, center + 20, y_error_bottom), fill=(25, 25, 25), width=3)

        draw.text((center, y_mean - 18), format_value(mean_value, y_max), font=bold_small, fill=colors["text"], anchor="mb")
        draw.text((center, plot_bottom + 30), CONDITION_LABELS[condition], font=axis_font, fill=colors["text"], anchor="ma")

    # Participant traces make the repeated-measures source visible without replacing the summary bars.
    subject_values: dict[str, list[tuple[int, float]]] = {}
    for condition in CONDITION_ORDER:
        for subject, value in scores.get((measure, panel, condition), []):
            if math.isnan(value):
                continue
            subject_values.setdefault(subject, []).append((centers[condition], value))
    for values in subject_values.values():
        if len(values) < 2:
            continue
        points = [(x, y_pos(value)) for x, value in values]
        draw.line(points, fill=colors["participant"], width=1)
        for x, y in points:
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=(20, 20, 20, 55))

    note_y = 750
    draw.text((plot_left, note_y), "Key planned comparisons:", font=bold_small, fill=colors["text"], anchor="la")
    for offset, comparison in enumerate(spec["comparisons"]):  # type: ignore[union-attr]
        row = pair_rows.get(str(comparison))
        if row is None:
            continue
        label = (
            f"{comparison}: mean diff={parse_float(row['mean_difference']):.2f}, "
            f"Holm {format_p(row['p_holm'])}, r={parse_float(row['effect_size']):.2f}"
        )
        draw.text((plot_left, note_y + 30 + offset * 30), label, font=small_font, fill=colors["text"], anchor="la")

    draw.text(
        (plot_left, height - 38),
        f"Source: stats_results.csv and participant_condition_scores.csv | {spec['discussion_point']}",
        font=small_font,
        fill=colors["muted"],
        anchor="la",
    )

    output_path = output_dir / str(spec["figure"])
    image.save(output_path)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize the statistical values cited in discussion_ko.txt.")
    parser.add_argument("--stats", type=Path, default=DEFAULT_STATS_PATH)
    parser.add_argument("--scores", type=Path, default=DEFAULT_SCORES_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--table-output", type=Path, default=DEFAULT_TABLE_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats_rows = read_csv(args.stats)
    score_rows = read_csv(args.scores)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    table_rows = build_source_table(stats_rows)
    write_csv(args.table_output, table_rows, TABLE_FIELDS)
    scores = score_lookup(score_rows)
    figure_paths = [plot_spec(spec, stats_rows, scores, args.output_dir) for spec in KEY_RESULTS]

    print(f"stats source: {args.stats}")
    print(f"score source: {args.scores}")
    print(f"source table: {args.table_output} ({len(table_rows)} rows)")
    print(f"figures: {args.output_dir} ({len(figure_paths)} files)")
    for path in figure_paths:
        print(f"- {path}")


if __name__ == "__main__":
    main()
