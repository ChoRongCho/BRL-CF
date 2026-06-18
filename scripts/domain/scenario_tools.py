#!/usr/bin/env python3
# Usage:
#   python3 scripts/domain/scenario_tools.py --mode 1 --seed 42 --overwrite  # assign balanced random labels in CSV
#   python3 scripts/domain/scenario_tools.py --mode 2                        # generate scene_06~15 YAML from CSV
#   python3 scripts/domain/scenario_tools.py --mode 3                        # print CSV scenario summary
#   python3 scripts/domain/scenario_tools.py --mode html                     # generate scenario_preview.html
#   python3 scripts/domain/scenario_tools.py --mode readme                   # update README.md with YAML scene summary
#   python3 scripts/domain/scenario_tools.py --mode all --overwrite          # run all generation/update steps
"""Generate, parse, and print domain scenario CSV files."""

from __future__ import annotations

import argparse
import csv
import random
import re
from html import escape
from collections import Counter
from pathlib import Path

import yaml


BASE_DIR = Path(__file__).resolve().parent
TOMATO_CSV = BASE_DIR / "tomato_scenario.csv"
WASTE_CSV = BASE_DIR / "waste_scenario.csv"
TOMATO_DIR = BASE_DIR / "tomato"
WASTE_DIR = BASE_DIR / "wastesorting"
PREVIEW_HTML = BASE_DIR / "scenario_preview.html"
README_PATH = BASE_DIR / "README.md"
README_START = "<!-- SCENARIO_SUMMARY_START -->"
README_END = "<!-- SCENARIO_SUMMARY_END -->"

GENERATE_RANGE = range(6, 16)
DISPLAY_RANGE = range(1, 16)
TOMATO_ATTRS = ("rp", "rt", "un")
WASTE_ATTRS = ("gw", "pp", "pl", "ca")
TOMATO_FACT = {"rp": "ripe", "rt": "rotten", "un": "unripe"}
TOMATO_GOAL = {
    "rp": lambda tomato, _stem: f"loaded({tomato}, brl_robot)",
    "rt": lambda tomato, _stem: f"discarded({tomato})",
    "un": lambda tomato, stem: f"at({tomato}, {stem})",
}
WASTE_FACT = {"gw": "general", "pp": "paper", "pl": "plastic", "ca": "can"}
WASTE_BIN = {"gw": "gw_bin", "pp": "pp_bin", "pl": "pl_bin", "ca": "ca_bin"}
TOMATO_LABEL = {"rp": "ripe", "rt": "rotten", "un": "unripe", "<blank>": "blank"}
WASTE_LABEL = {"gw": "general", "pp": "paper", "pl": "plastic", "ca": "can", "<blank>": "blank"}


def read_csv(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open(newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        return header, list(reader)


def write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def scenario_number(row: list[str]) -> int | None:
    match = re.fullmatch(r"Scenario(\d+)", row[0].strip())
    return int(match.group(1)) if match else None


def is_active_cell(value: str) -> bool:
    return value.strip().upper() != "N/A"


def balanced_random_attrs(attrs: tuple[str, ...], count: int, rng: random.Random) -> list[str]:
    base, remainder = divmod(count, len(attrs))
    values = [attr for attr in attrs for _ in range(base)]
    extras = list(attrs)
    rng.shuffle(extras)
    values.extend(extras[:remainder])
    rng.shuffle(values)
    return values


def active_tomato_slots(row: list[str]) -> list[int]:
    return [idx for idx in range(1, 17) if is_active_cell(row[idx])]


def active_waste_slots(row: list[str]) -> list[int]:
    return [idx for idx in range(1, 21) if is_active_cell(row[idx])]


def mode_1_assign(seed: int, overwrite: bool) -> None:
    rng = random.Random(seed)

    tomato_header, tomato_rows = read_csv(TOMATO_CSV)
    for row in tomato_rows:
        number = scenario_number(row)
        if number not in GENERATE_RANGE:
            continue
        slots = active_tomato_slots(row)
        attrs = balanced_random_attrs(TOMATO_ATTRS, len(slots), rng)
        for idx, attr in zip(slots, attrs):
            if overwrite or not row[idx].strip():
                row[idx] = attr
    write_csv(TOMATO_CSV, tomato_header, tomato_rows)

    waste_header, waste_rows = read_csv(WASTE_CSV)
    for row in waste_rows:
        number = scenario_number(row)
        if number not in GENERATE_RANGE:
            continue
        slots = active_waste_slots(row)
        attrs = balanced_random_attrs(WASTE_ATTRS, len(slots), rng)
        for idx, attr in zip(slots, attrs):
            if overwrite or not row[idx].strip():
                row[idx] = attr
    write_csv(WASTE_CSV, waste_header, waste_rows)


def checked_attr(value: str, valid: tuple[str, ...], scenario: int, label: str, strict: bool) -> str:
    attr = value.strip()
    if attr not in valid:
        if not strict and not attr:
            return "<blank>"
        raise ValueError(f"Scenario{scenario} {label} has invalid or empty attr: {value!r}")
    return attr


def parse_tomato_row(row: list[str], strict: bool = True) -> list[tuple[str, str, str]]:
    number = scenario_number(row)
    if number is None:
        return []

    tomatoes: list[tuple[str, str, str]] = []
    for idx in active_tomato_slots(row):
        attr = checked_attr(row[idx], TOMATO_ATTRS, number, f"col {idx}", strict)
        stem_index = (idx - 1) // 4 + 1
        stem = f"stem_{stem_index:02d}"
        tomato = f"tomato{len(tomatoes) + 1}"
        tomatoes.append((tomato, attr, stem))
    return tomatoes


def parse_waste_row(row: list[str], strict: bool = True) -> tuple[list[tuple[str, str]], list[tuple[int, int]]]:
    number = scenario_number(row)
    if number is None:
        return [], []

    wastes: list[tuple[str, str]] = []
    for idx in active_waste_slots(row):
        attr = checked_attr(row[idx], WASTE_ATTRS, number, f"w{idx}", strict)
        wastes.append((f"waste{len(wastes) + 1}", attr))

    on_text = row[21] if len(row) > 21 else ""
    relations = []
    for lower, upper in re.findall(r"on\((\d+)\s*,\s*(\d+)\)", on_text, flags=re.IGNORECASE):
        a, b = int(lower), int(upper)
        if a <= len(wastes) and b <= len(wastes):
            relations.append((a, b))
    return wastes, relations


def write_yaml(path: Path, domain: str, description: str, types: list[tuple[str, list[str]]],
               facts: list[str], true_init: list[str], goal: list[str]) -> None:
    lines = [
        f"domain: {domain}",
        "",
        "meta_data:",
        f'  description: "{description}"',
        '  purpose: "Runtime facts for possible world generation"',
        "",
        "",
        "type:",
    ]
    for type_name, values in types:
        lines.append(f"  {type_name}:")
        lines.extend(f"  - {value}" for value in values)

    lines.extend(["", "", "facts:"])
    lines.extend(f'  - "{fact}"' for fact in facts)
    lines.extend(["", "", "true_init:"])
    lines.extend(f'  - "{fact}"' for fact in true_init)
    lines.extend(["", "", "goal:"])
    lines.extend(f'  - "{fact}"' for fact in goal)
    path.write_text("\n".join(lines) + "\n")


def tomato_yaml(row: list[str], output_dir: Path) -> None:
    number = scenario_number(row)
    if number is None or number not in GENERATE_RANGE:
        return
    tomatoes = parse_tomato_row(row)

    stems = sorted({stem for _tomato, _attr, stem in tomatoes})
    tomato_names = [tomato for tomato, _attr, _stem in tomatoes]
    facts = [f"tomato({tomato})" for tomato in tomato_names]
    facts.extend(["robot(brl_robot)", "location(dock_station)"])
    facts.extend(f"location({stem})" for stem in stems)
    facts.append("handempty(brl_robot)")
    facts.extend(f"stem({stem})" for stem in stems)
    facts.append("located(brl_robot, dock_station)")

    true_init = [f"{TOMATO_FACT[attr]}({tomato})" for tomato, attr, _stem in tomatoes]
    true_init.extend(f"at({tomato}, {stem})" for tomato, _attr, stem in tomatoes)

    goal = [f"observed({tomato})" for tomato in tomato_names]
    goal.extend(TOMATO_GOAL[attr](tomato, stem) for tomato, attr, stem in tomatoes)

    types = [
        ("tomato(T)", tomato_names),
        ("robot(R)", ["brl_robot"]),
        ("location(L)", ["dock_station", *stems]),
        ("stem(S)", stems),
    ]
    write_yaml(
        output_dir / f"scene_{number:02d}.yaml",
        "TomatoHarvest",
        "Initial state facts for tomato harvesting domain",
        types,
        facts,
        true_init,
        goal,
    )


def waste_yaml(row: list[str], output_dir: Path) -> None:
    number = scenario_number(row)
    if number is None or number not in GENERATE_RANGE:
        return
    wastes, relations = parse_waste_row(row)

    waste_names = [waste for waste, _attr in wastes]
    facts = [f"waste({waste})" for waste in waste_names]
    facts.extend([
        "robot(brl_robot)",
        "plastic_bin(pl_bin)",
        "can_bin(ca_bin)",
        "paper_bin(pp_bin)",
        "general_bin(gw_bin)",
        "handempty(brl_robot)",
    ])

    true_init = [f"{WASTE_FACT[attr]}({waste})" for waste, attr in wastes]
    true_init.extend(f"on(waste{a}, waste{b})" for a, b in relations)

    goal = [f"in_bin({waste}, {WASTE_BIN[attr]})" for waste, attr in wastes]
    types = [
        ("waste(W)", waste_names),
        ("robot(R)", ["brl_robot"]),
        ("plastic_bin(B)", ["pl_bin"]),
        ("can_bin(B)", ["ca_bin"]),
        ("paper_bin(B)", ["pp_bin"]),
        ("general_bin(B)", ["gw_bin"]),
    ]
    write_yaml(
        output_dir / f"scene_{number:02d}.yaml",
        "WasteSorting",
        "Initial state facts for waste sorting domain",
        types,
        facts,
        true_init,
        goal,
    )


def mode_2_parse_yaml() -> None:
    _header, tomato_rows = read_csv(TOMATO_CSV)
    TOMATO_DIR.mkdir(parents=True, exist_ok=True)
    for row in tomato_rows:
        tomato_yaml(row, TOMATO_DIR)

    _header, waste_rows = read_csv(WASTE_CSV)
    WASTE_DIR.mkdir(parents=True, exist_ok=True)
    for row in waste_rows:
        waste_yaml(row, WASTE_DIR)


def print_counter(counter: Counter[str], order: tuple[str, ...]) -> str:
    return ", ".join(f"{attr}={counter.get(attr, 0)}" for attr in order)


def mode_3_print() -> None:
    _header, tomato_rows = read_csv(TOMATO_CSV)
    print("[TomatoHarvest]")
    for row in tomato_rows:
        number = scenario_number(row)
        if number not in DISPLAY_RANGE:
            continue
        tomatoes = parse_tomato_row(row, strict=False)
        counts = Counter(attr for _tomato, attr, _stem in tomatoes)
        stems: dict[str, list[str]] = {}
        for tomato, attr, stem in tomatoes:
            stems.setdefault(stem, []).append(f"{tomato}:{attr}")
        stem_text = "; ".join(f"{stem}({', '.join(values)})" for stem, values in stems.items())
        print(f"  Scenario{number}: {len(tomatoes)} tomatoes | {print_counter(counts, TOMATO_ATTRS)} | {stem_text}")

    _header, waste_rows = read_csv(WASTE_CSV)
    print("\n[WasteSorting]")
    for row in waste_rows:
        number = scenario_number(row)
        if number not in DISPLAY_RANGE:
            continue
        wastes, relations = parse_waste_row(row, strict=False)
        counts = Counter(attr for _waste, attr in wastes)
        rel_text = ", ".join(f"on({a},{b})" for a, b in relations) or "-"
        waste_text = ", ".join(f"{waste}:{attr}" for waste, attr in wastes)
        print(f"  Scenario{number}: {len(wastes)} wastes | {print_counter(counts, WASTE_ATTRS)} | {waste_text} | {rel_text}")


def css_class(attr: str) -> str:
    return {
        "rp": "ripe",
        "rt": "rotten",
        "un": "unripe",
        "gw": "general",
        "pp": "paper",
        "pl": "plastic",
        "ca": "can",
    }.get(attr, "blank")


def tomato_card(row: list[str]) -> str:
    number = scenario_number(row)
    tomatoes = parse_tomato_row(row, strict=False)
    counts = Counter(attr for _tomato, attr, _stem in tomatoes)
    stems: dict[str, list[tuple[str, str]]] = {}
    for tomato, attr, stem in tomatoes:
        stems.setdefault(stem, []).append((tomato, attr))

    stem_html = []
    for stem in sorted(stems):
        cells = []
        for tomato, attr in stems[stem]:
            label = TOMATO_LABEL.get(attr, attr)
            cells.append(
                f'<div class="chip tomato {css_class(attr)}">'
                f'<strong>{escape(tomato.replace("tomato", "t"))}</strong><span>{escape(label)}</span></div>'
            )
        stem_html.append(
            f'<section class="stem"><h4>{escape(stem)}</h4><div class="chip-grid">{"".join(cells)}</div></section>'
        )

    return (
        f'<article class="scenario-row"><header><h3>Scenario {number}</h3>'
        f'<p>{len(tomatoes)} tomatoes<br>{escape(print_counter(counts, TOMATO_ATTRS))}</p></header>'
        f'<div class="stem-grid">{"".join(stem_html)}</div></article>'
    )


def waste_positions(count: int, relations: list[tuple[int, int]]) -> tuple[dict[int, tuple[int, int]], int, int]:
    children: dict[int, list[int]] = {idx: [] for idx in range(1, count + 1)}
    supported_by: dict[int, int] = {}
    for upper, lower in relations:
        if upper == lower or upper < 1 or lower < 1 or upper > count or lower > count:
            continue
        children.setdefault(lower, []).append(upper)
        supported_by[upper] = lower

    depths: dict[int, int] = {}

    def depth(idx: int, visiting: set[int] | None = None) -> int:
        visiting = visiting or set()
        if idx in depths:
            return depths[idx]
        if idx in visiting or idx not in supported_by:
            depths[idx] = 0
            return 0
        visiting.add(idx)
        depths[idx] = depth(supported_by[idx], visiting) + 1
        visiting.remove(idx)
        return depths[idx]

    for idx in range(1, count + 1):
        depth(idx)

    positions: dict[int, tuple[int, int]] = {}
    leaf_gap = 86
    level_gap = 58
    margin_x = 56
    max_depth = max(depths.values(), default=0)
    table_y = 62 + max_depth * level_gap
    next_x = margin_x

    def place_tree(idx: int) -> float:
        nonlocal next_x
        branch_children = sorted(children.get(idx, []))
        if not branch_children:
            x = next_x
            next_x += leaf_gap
        else:
            child_xs = [place_tree(child) for child in branch_children]
            x = sum(child_xs) / len(child_xs)
        positions[idx] = (round(x), table_y - depths[idx] * level_gap)
        return x

    for bottom in [idx for idx in range(1, count + 1) if idx not in supported_by]:
        place_tree(bottom)

    width = max(520, round(next_x + margin_x - leaf_gap))
    height = table_y + 44
    return positions, width, height


def unique_support_relations(relations: list[tuple[int, int]]) -> list[tuple[int, int]]:
    seen_upper: set[int] = set()
    unique = []
    for upper, lower in relations:
        if upper in seen_upper:
            continue
        seen_upper.add(upper)
        unique.append((upper, lower))
    return unique


def waste_card(row: list[str]) -> str:
    number = scenario_number(row)
    wastes, relations = parse_waste_row(row, strict=False)
    layout_relations = unique_support_relations(relations)
    counts = Counter(attr for _waste, attr in wastes)
    radius = 22
    positions, width, height = waste_positions(len(wastes), layout_relations)
    table_y = max(y for _x, y in positions.values()) if positions else 188

    edge_svg = []
    for upper, lower in layout_relations:
        if upper not in positions or lower not in positions:
            continue
        x1, y1 = positions[upper]
        x2, y2 = positions[lower]
        edge_svg.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" marker-end="url(#arrow)" />'
        )

    node_svg = []
    for idx, (_waste, attr) in enumerate(wastes, start=1):
        x, y = positions[idx]
        node_svg.append(
            f'<g class="node {css_class(attr)}"><circle cx="{x}" cy="{y}" r="{radius}" />'
            f'<text x="{x}" y="{y - 1}">w{idx}</text><text x="{x}" y="{y + 28}">{escape(attr)}</text></g>'
        )

    chips = []
    for waste, attr in wastes:
        label = WASTE_LABEL.get(attr, attr)
        chips.append(
            f'<div class="chip waste {css_class(attr)}"><strong>{escape(waste)}</strong><span>{escape(label)}</span></div>'
        )

    rel_text = ", ".join(f"on({a},{b})" for a, b in relations) or "-"
    ignored_count = len(relations) - len(layout_relations)
    if ignored_count:
        rel_text = f"{rel_text} | display ignored duplicate support: {ignored_count}"
    return (
        f'<article class="scenario-row waste-row"><header><h3>Scenario {number}</h3>'
        f'<p>{len(wastes)} wastes<br>{escape(print_counter(counts, WASTE_ATTRS))}</p></header>'
        '<div class="waste-stage">'
        f'<svg class="waste-graph" viewBox="0 0 {width} {height}" role="img">'
        '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto">'
        '<path d="M0,0 L0,6 L7,3 z"></path></marker></defs>'
        f'<line class="table-line" x1="18" y1="{table_y + radius}" x2="{width - 18}" y2="{table_y + radius}" />'
        f'<text class="table-label" x="22" y="{table_y + radius + 24}">table</text>'
        f'<g class="edges">{"".join(edge_svg)}</g><g>{"".join(node_svg)}</g></svg>'
        '</div>'
        f'<div class="waste-side"><p class="relations">{escape(rel_text)}</p><div class="waste-grid">{"".join(chips)}</div></div></article>'
    )


def mode_html_preview() -> None:
    _header, tomato_rows = read_csv(TOMATO_CSV)
    _header, waste_rows = read_csv(WASTE_CSV)
    tomato_cards = []
    waste_cards = []
    for row in tomato_rows:
        number = scenario_number(row)
        if number in DISPLAY_RANGE:
            tomato_cards.append(tomato_card(row))
    for row in waste_rows:
        number = scenario_number(row)
        if number in DISPLAY_RANGE:
            waste_cards.append(waste_card(row))

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Scenario Preview</title>
<style>
:root {{
  color-scheme: light;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  color: #17202a;
  background: #f6f7f9;
}}
body {{ margin: 0; padding: 28px; }}
h1 {{ margin: 0 0 8px; font-size: 28px; }}
h2 {{ margin: 34px 0 14px; font-size: 22px; }}
h3 {{ margin: 0; font-size: 17px; }}
h4 {{ margin: 0 0 10px; font-size: 14px; color: #4b5563; }}
p {{ margin: 4px 0 0; color: #5b6573; }}
.legend {{ display: flex; flex-wrap: wrap; gap: 8px; margin: 14px 0 22px; }}
.section-list {{ display: grid; gap: 10px; }}
.scenario-row {{ display: grid; grid-template-columns: 132px minmax(0, 1fr); gap: 14px; align-items: center; background: white; border: 1px solid #dfe3e8; border-radius: 8px; padding: 12px; overflow-x: auto; }}
.waste-row {{ grid-template-columns: 132px minmax(520px, 1fr) 300px; align-items: start; }}
.scenario-row header {{ min-width: 120px; }}
.waste-stage {{ min-width: 520px; border: 1px solid #eef1f4; border-radius: 8px; background: #fbfcfd; padding: 8px 12px 4px; }}
.waste-side {{ min-width: 260px; max-height: 260px; overflow: auto; padding: 2px 4px; }}
.stem-grid {{ display: flex; gap: 10px; min-width: max-content; }}
.stem {{ border: 1px solid #e5e8ec; border-radius: 8px; padding: 10px; background: #fbfcfd; min-width: 150px; }}
.chip-grid, .waste-grid {{ display: flex; flex-wrap: wrap; gap: 6px; }}
.waste-grid {{ display: grid; grid-template-columns: repeat(4, minmax(58px, 1fr)); align-items: start; }}
.chip {{ border-radius: 8px; padding: 6px 8px; border: 1px solid transparent; min-height: 34px; min-width: 58px; }}
.chip strong {{ display: block; font-size: 13px; }}
.chip span {{ display: block; font-size: 12px; margin-top: 2px; }}
.ripe {{ background: #e8f7ef; border-color: #93d5ad; color: #14532d; }}
.rotten {{ background: #fdecec; border-color: #f2a3a3; color: #7f1d1d; }}
.unripe {{ background: #fff7d6; border-color: #e7c95a; color: #6f5200; }}
.general {{ background: #eef1f4; border-color: #b9c0ca; color: #374151; }}
.paper {{ background: #e8f1ff; border-color: #9dc0f5; color: #1e3a8a; }}
.plastic {{ background: #e8fbf8; border-color: #8fd8cf; color: #115e59; }}
.can {{ background: #f3ebff; border-color: #c6a7f2; color: #581c87; }}
.blank {{ background: #f8fafc; border-color: #cbd5e1; color: #64748b; }}
.waste-graph {{ display: block; width: 100%; min-width: 520px; max-height: 320px; height: auto; margin: 0; }}
.edges line {{ stroke: #687385; stroke-width: 2; opacity: .78; }}
marker path {{ fill: #687385; }}
.node circle {{ stroke-width: 2; }}
.node text {{ text-anchor: middle; font-size: 12px; font-weight: 800; fill: currentColor; }}
.table-line {{ stroke: #2f3542; stroke-width: 3; }}
.table-label {{ font-size: 12px; fill: #5b6573; font-weight: 700; }}
.relations {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; line-height: 1.35; margin: 0 0 8px; color: #334155; }}
@media (max-width: 760px) {{
  body {{ padding: 16px; }}
  .scenario-row {{ grid-template-columns: 1fr; }}
  .waste-row {{ grid-template-columns: 1fr; }}
}}
</style>
</head>
<body>
<h1>Scenario Preview</h1>
<p>Generated from tomato_scenario.csv and waste_scenario.csv.</p>
<div class="legend">
  <div class="chip tomato ripe"><strong>rp</strong><span>ripe</span></div>
  <div class="chip tomato rotten"><strong>rt</strong><span>rotten</span></div>
  <div class="chip tomato unripe"><strong>un</strong><span>unripe</span></div>
  <div class="chip waste general"><strong>gw</strong><span>general</span></div>
  <div class="chip waste paper"><strong>pp</strong><span>paper</span></div>
  <div class="chip waste plastic"><strong>pl</strong><span>plastic</span></div>
  <div class="chip waste can"><strong>ca</strong><span>can</span></div>
</div>
<h2>Tomato Harvest</h2>
<section class="section-list">
{"".join(tomato_cards)}
</section>
<h2>Waste Sorting</h2>
<section class="section-list">
{"".join(waste_cards)}
</section>
</body>
</html>
"""
    PREVIEW_HTML.write_text(html)
    print(PREVIEW_HTML)


def scene_number_from_path(path: Path) -> int:
    match = re.fullmatch(r"scene_(\d+)\.yaml", path.name)
    return int(match.group(1)) if match else -1


def yaml_facts(path: Path) -> tuple[dict, list[str]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return data, list(data.get("true_init", []) or [])


def count_type(data: dict, type_name: str) -> int:
    return len(data.get("type", {}).get(type_name, []) or [])


def fact_counts(facts: list[str], names: tuple[str, ...]) -> Counter[str]:
    counts: Counter[str] = Counter()
    pattern = re.compile(r"^(" + "|".join(re.escape(name) for name in names) + r")\(")
    for fact in facts:
        match = pattern.match(fact)
        if match:
            counts[match.group(1)] += 1
    return counts


def tomato_distribution(facts: list[str]) -> str:
    stems: dict[str, list[str]] = {}
    for fact in facts:
        match = re.fullmatch(r"at\((tomato\d+),\s*(stem_\d+)\)", fact)
        if match:
            stems.setdefault(match.group(2), []).append(match.group(1))
    if not stems:
        return "-"
    return "/".join(str(len(stems[stem])) for stem in sorted(stems, key=natural_key))


def tomato_stem_detail(facts: list[str]) -> str:
    labels: dict[str, str] = {}
    stems: dict[str, list[str]] = {}
    for fact in facts:
        label_match = re.fullmatch(r"(ripe|rotten|unripe)\((tomato\d+)\)", fact)
        at_match = re.fullmatch(r"at\((tomato\d+),\s*(stem_\d+)\)", fact)
        if label_match:
            labels[label_match.group(2)] = label_match.group(1)
        elif at_match:
            stems.setdefault(at_match.group(2), []).append(at_match.group(1))
    chunks = []
    for stem in sorted(stems, key=natural_key):
        values = []
        for tomato in sorted(stems[stem], key=natural_key):
            values.append(f"{tomato.replace('tomato', 't')}:{labels.get(tomato, '?')}")
        chunks.append(f"{stem}({', '.join(values)})")
    return "; ".join(chunks) or "-"


def waste_on_detail(facts: list[str]) -> str:
    relations = []
    for fact in facts:
        match = re.fullmatch(r"on\(waste(\d+),\s*waste(\d+)\)", fact)
        if match:
            relations.append((int(match.group(1)), int(match.group(2))))
    if not relations:
        return "-"
    return ", ".join(f"on({a},{b})" for a, b in sorted(relations))


def natural_key(text: str) -> tuple[str, int, str]:
    match = re.search(r"(\d+)", text)
    if not match:
        return text, -1, text
    return text[: match.start()], int(match.group(1)), text[match.end() :]


def markdown_counter(counter: Counter[str], order: tuple[str, ...]) -> str:
    return ", ".join(f"{name}={counter.get(name, 0)}" for name in order)


def tomato_readme_rows() -> list[str]:
    rows = [
        "| Scene | Tomatoes | Stems | Distribution | Labels | Placement |",
        "|---:|---:|---:|---|---|---|",
    ]
    for path in sorted(TOMATO_DIR.glob("scene_*.yaml"), key=scene_number_from_path):
        data, facts = yaml_facts(path)
        scene = scene_number_from_path(path)
        labels = fact_counts(facts, ("ripe", "rotten", "unripe"))
        rows.append(
            "| "
            f"{scene:02d} | "
            f"{count_type(data, 'tomato(T)')} | "
            f"{count_type(data, 'stem(S)')} | "
            f"{tomato_distribution(facts)} | "
            f"{markdown_counter(labels, ('ripe', 'rotten', 'unripe'))} | "
            f"{tomato_stem_detail(facts)} |"
        )
    return rows


def waste_readme_rows() -> list[str]:
    rows = [
        "| Scene | Wastes | Labels | Occlusion |",
        "|---:|---:|---|---|",
    ]
    for path in sorted(WASTE_DIR.glob("scene_*.yaml"), key=scene_number_from_path):
        data, facts = yaml_facts(path)
        scene = scene_number_from_path(path)
        labels = fact_counts(facts, ("general", "paper", "plastic", "can"))
        rows.append(
            "| "
            f"{scene:02d} | "
            f"{count_type(data, 'waste(W)')} | "
            f"{markdown_counter(labels, ('general', 'paper', 'plastic', 'can'))} | "
            f"{waste_on_detail(facts)} |"
        )
    return rows


def scenario_readme_content() -> str:
    lines = [
        "# Domain Scenarios",
        "",
        "This file is generated from `scripts/domain/{tomato,wastesorting}/scene_XX.yaml`.",
        "",
        README_START,
        "",
        "## Tomato Harvest",
        "",
        *tomato_readme_rows(),
        "",
        "## Waste Sorting",
        "",
        *waste_readme_rows(),
        "",
        README_END,
        "",
    ]
    return "\n".join(lines)


def mode_readme_summary() -> None:
    generated = scenario_readme_content()
    current = README_PATH.read_text(encoding="utf-8") if README_PATH.exists() else ""
    if README_START in current and README_END in current:
        prefix = current.split(README_START, 1)[0].rstrip()
        suffix = current.split(README_END, 1)[1].lstrip()
        summary = generated.split(README_START, 1)[1].split(README_END, 1)[0]
        next_text = f"{prefix}\n\n{README_START}{summary}{README_END}\n"
        if suffix:
            next_text += f"\n{suffix}"
    else:
        next_text = generated
    README_PATH.write_text(next_text, encoding="utf-8")
    print(README_PATH)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scenario CSV/YAML helper for scripts/domain.")
    parser.add_argument("--mode", required=True, choices=("1", "2", "3", "html", "readme", "all"))
    parser.add_argument("--seed", type=int, default=42, help="Random seed for mode 1.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite non-empty scenario attrs in mode 1.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode in ("1", "all"):
        mode_1_assign(args.seed, args.overwrite)
    if args.mode in ("2", "all"):
        mode_2_parse_yaml()
    if args.mode in ("3", "all"):
        mode_3_print()
    if args.mode in ("html", "all"):
        mode_html_preview()
    if args.mode in ("readme", "all"):
        mode_readme_summary()


if __name__ == "__main__":
    main()
