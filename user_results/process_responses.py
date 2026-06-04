from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_RESPONSE_PATH = BASE_DIR / "response.csv"
DEFAULT_ANSWER_PATH = BASE_DIR / "answer_sheet.csv"
DEFAULT_OUTPUT_PATH = BASE_DIR / "output.csv"
DEFAULT_METADATA_OUTPUT_PATH = BASE_DIR / "metadata_summary.txt"

KEY_FIELDS = ("평가 도메인", "시나리오", "조건 선택")
QUESTIONS = ("Q1", "Q2", "Q3", "Q4", "Q5", "Q6")
SCENARIO_ORDER = ("시나리오1", "시나리오2", "시나리오3", "시나리오4", "시나리오5")
CONDITION_ORDER = ("C1: All", "C2: No", "C3: Ours1", "C4: Ours2", "C5: KnowNo")
DOMAIN_ORDER = ("토마토 수확", "분리수거")
DOMAIN_PANELS: tuple[tuple[str, str | None], ...] = (
    ("토마토 수확", "토마토 수확"),
    ("분리수거", "분리수거"),
    ("통합", None),
)
SCALE_MIN = 1.0
SCALE_MAX = 7.0
REVERSE_CODED_ITEMS: tuple[str, ...] = ()

DOMAIN_Q_COLUMNS = {
    "토마토 수확": {
        "Q1": "Q1. 현재 화면에 표시된 토마토는 총 몇개입니까?",
        "Q2": "Q2. 로봇이 마지막으로 처리한 토마토의 상태는 무엇입니까?",
        "Q3": "Q3. 현재 상황에서 사용자가 판단해야 하는 핵심 정보는 무엇입니까?",
        "Q4": "Q4. 현재 상황에서 사용자의 확인이 필요한 이유는 무엇입니까?",
        "Q5": "Q5. 현재 사용자가 아무 행동도 하지 않으면 로봇은 어떻게 될 가능성이 가장 높습니까?",
        "Q6": "Q6. 로봇이 잘못된 토마토를 선택했는데 사용자가 수정하지 않으면, 어떤 결과가 발생할 가능성이 가장 높습니까?",
    },
    "분리수거": {
        "Q1": "Q1. 현재 화면에서 테이블 위에 놓여있는 물건은 총 몇개입니까?",
        "Q2": "Q2. 로봇이 마지막으로 처리한 쓰레기의 재질은 무엇입니까?",
        "Q3": "Q3. 현재 상황에서 사용자가 판단해야 하는 핵심 정보는 무엇입니까?",
        "Q4": "Q4. 방금 상황에서 사용자의 확인이 필요한 이유는 무엇입니까?",
        "Q5": "Q5. 현재 사용자가 아무 행동도 하지 않으면 로봇은 어떻게 될 가능성이 가장 높습니까?",
        "Q6": "Q6. 로봇이 잘못된 분리수거를 했는데 사용자가 작업을 수정하지 않으면, 어떤 결과가 발생할 가능성이 가장 높습니까?",
    },
}

SAGAT_ITEM_TITLES = {
    "Q1": "Q1. 현재 화면의 대상 물체 개수",
    "Q2": "Q2. 로봇이 마지막으로 처리한 대상의 상태/재질",
    "Q3": "Q3. 현재 상황에서 사용자가 판단해야 하는 핵심 정보",
    "Q4": "Q4. 현재 상황에서 사용자의 확인이 필요한 이유",
    "Q5": "Q5. 사용자가 아무 행동도 하지 않을 때 예상되는 로봇 행동",
    "Q6": "Q6. 잘못된 로봇 작업을 수정하지 않을 때 예상되는 결과",
}

OUTPUT_FIELDS = [
    "척도",
    "문항",
    "문항내용",
    "도메인패널",
    "조건 선택",
    "평균",
    "표준편차",
    "표본수",
    "값범위최소",
    "값범위최대",
]


def normalize(value: str | None) -> str:
    return " ".join((value or "").strip().split())


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def row_key(row: dict[str, str]) -> tuple[str, str, str]:
    return tuple(normalize(row.get(field)) for field in KEY_FIELDS)


def build_answer_lookup(answer_rows: list[dict[str, str]]) -> dict[tuple[str, str, str], dict[str, str]]:
    return {
        row_key(row): {question: normalize(row.get(question)) for question in QUESTIONS}
        for row in answer_rows
    }


def score_sagat_rows(
    response_rows: list[dict[str, str]],
    answer_lookup: dict[tuple[str, str, str], dict[str, str]],
) -> list[dict[str, str]]:
    item_rows: list[dict[str, str]] = []

    for response in response_rows:
        subject = normalize(response.get("피험자"))
        domain = normalize(response.get("평가 도메인"))
        scenario = normalize(response.get("시나리오"))
        condition = normalize(response.get("조건 선택"))
        question_columns = DOMAIN_Q_COLUMNS.get(domain)
        answers = answer_lookup.get((domain, scenario, condition))

        for question in QUESTIONS:
            if question_columns is None or answers is None:
                continue

            answer_value = answers.get(question, "")
            response_value = normalize(response.get(question_columns[question]))
            if not answer_value or not response_value:
                continue

            item_rows.append(
                {
                    "척도": "SAGAT",
                    "피험자": subject,
                    "문항": question,
                    "문항내용": SAGAT_ITEM_TITLES[question],
                    "평가 도메인": domain,
                    "조건 선택": condition,
                    "값": "1" if response_value == answer_value else "0",
                }
            )

    return item_rows


def format_rate(correct: int, total: int) -> str:
    if total == 0:
        return "NA"
    return f"{correct / total:.3f}"


def item_id_from_scale_column(column: str) -> str | None:
    item_id = column.split(".", 1)[0].strip()
    if item_id == "F":
        return item_id
    if len(item_id) >= 2 and item_id[0] in {"N", "S"} and item_id[1:].isdigit():
        return item_id
    return None


def scale_question_sort_key(item_id: str) -> tuple[int, int]:
    if item_id.startswith("N"):
        return (0, int(item_id[1:]))
    if item_id == "F":
        return (1, 0)
    if item_id.startswith("S"):
        return (2, int(item_id[1:]))
    return (3, 0)


def scale_question_columns(response_rows: list[dict[str, str]]) -> dict[str, str]:
    if not response_rows:
        return {}

    columns: dict[str, str] = {}
    for column in response_rows[0]:
        item_id = item_id_from_scale_column(column)
        if item_id is not None:
            columns[item_id] = column
    return dict(sorted(columns.items(), key=lambda item: scale_question_sort_key(item[0])))


def parse_float(value: str | None) -> float | None:
    normalized = normalize(value)
    if not normalized:
        return None
    try:
        return float(normalized)
    except ValueError:
        return None


def coded_scale_value(item_id: str, value: float | None) -> float | None:
    if value is None:
        return None
    if item_id in REVERSE_CODED_ITEMS:
        return SCALE_MIN + SCALE_MAX - value
    return value


def read_scale_rows(response_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    item_rows: list[dict[str, str]] = []
    scale_columns = scale_question_columns(response_rows)
    for response in response_rows:
        domain = normalize(response.get("평가 도메인"))
        condition = normalize(response.get("조건 선택"))

        for item_id, column in scale_columns.items():
            value = coded_scale_value(item_id, parse_float(response.get(column)))
            if value is None:
                continue
            item_rows.append(
                {
                    "척도": "Survey",
                    "문항": item_id,
                    "문항내용": column,
                    "평가 도메인": domain,
                    "조건 선택": condition,
                    "값": f"{value:g}",
                }
            )

    return item_rows


def condition_value(condition: str) -> str:
    for expected_condition in CONDITION_ORDER:
        if condition == expected_condition or condition.startswith(expected_condition.split(":", 1)[0]):
            return expected_condition
    return condition


def condition_code(condition: str) -> str:
    match = re.search(r"C\s*(\d+)", condition)
    if match:
        return f"C{match.group(1)}"
    return condition.replace(" ", "")


def scenario_code(scenario: str) -> str:
    match = re.search(r"(\d+)", scenario)
    if match:
        return f"S{match.group(1)}"
    return scenario.replace(" ", "")


def subject_code(subject: str) -> str:
    match = re.search(r"\d+", subject)
    if match:
        return f"p{int(match.group(0)):03d}"
    return subject or "unknown"


def domain_code(domain: str) -> str:
    if domain == "토마토 수확":
        return "Tomato"
    if domain == "분리수거":
        return "WasteSorting"
    return domain or "UnknownDomain"


def domain_sequence_code(domain: str) -> str:
    if domain == "토마토 수확":
        return "T"
    if domain == "분리수거":
        return "R"
    return (domain or "UNK").replace(" ", "")


def parse_timestamp(value: str) -> tuple[tuple[int, int, int, int, int, int, int], str]:
    normalized = normalize(value)
    match = re.match(
        r"(\d{4})\.\s*(\d{1,2})\.\s*(\d{1,2})\s*(오전|오후)\s*(\d{1,2}):(\d{2}):(\d{2})",
        normalized,
    )
    if not match:
        return (9999, 12, 31, 23, 59, 59, 0), normalized or "unknown"

    year, month, day, meridiem, hour, minute, second = match.groups()
    hour_int = int(hour)
    if meridiem == "오전" and hour_int == 12:
        hour_int = 0
    elif meridiem == "오후" and hour_int != 12:
        hour_int += 12

    key = (int(year), int(month), int(day), hour_int, int(minute), int(second), 0)
    date_label = f"{int(year):04d}-{int(month):02d}-{int(day):02d}"
    return key, date_label


def row_sort_key(row: dict[str, str]) -> tuple[int, int, int, int, int, int, int]:
    return parse_timestamp(row.get("타임스탬프", ""))[0]


def expected_scenarios(response_rows: list[dict[str, str]]) -> list[str]:
    scenarios = {normalize(row.get("시나리오")) for row in response_rows if normalize(row.get("시나리오"))}
    ordered = list(SCENARIO_ORDER)
    ordered.extend(
        sorted(
            scenarios - set(ordered),
            key=lambda value: int(re.search(r"\d+", value).group(0)) if re.search(r"\d+", value) else 999,
        )
    )
    return ordered


def expected_domains(response_rows: list[dict[str, str]]) -> list[str]:
    observed = {normalize(row.get("평가 도메인")) for row in response_rows if normalize(row.get("평가 도메인"))}
    ordered = list(DOMAIN_ORDER)
    ordered.extend(sorted(observed - set(ordered)))
    return ordered


def build_participant_accuracy_lines(
    subjects: list[str],
    sagat_rows: list[dict[str, str]],
) -> list[str]:
    rows_by_subject: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in sagat_rows:
        rows_by_subject[row.get("피험자", "")].append(row)

    lines: list[str] = ["", "Participant SAGAT accuracy", "--------------------------"]
    for subject in subjects:
        subject_rows = rows_by_subject.get(subject, [])
        parts: list[str] = []
        total_correct = 0
        total_scored = 0
        for question in QUESTIONS:
            question_rows = [row for row in subject_rows if row["문항"] == question]
            scored = len(question_rows)
            correct = sum(1 for row in question_rows if row["값"] == "1")
            total_correct += correct
            total_scored += scored
            parts.append(f"{question}={format_rate(correct, scored)}({correct}/{scored})")
        parts.append(f"Total={format_rate(total_correct, total_scored)}({total_correct}/{total_scored})")
        lines.append(f"{subject_code(subject)}: " + ", ".join(parts))
    return lines


def build_metadata_summary(response_rows: list[dict[str, str]], sagat_rows: list[dict[str, str]]) -> str:
    subjects = sorted(
        {normalize(row.get("피험자")) for row in response_rows if normalize(row.get("피험자"))},
        key=lambda value: int(re.search(r"\d+", value).group(0)) if re.search(r"\d+", value) else 999999,
    )
    sorted_rows = sorted(response_rows, key=row_sort_key)
    timestamp_rows = [row for row in sorted_rows if normalize(row.get("타임스탬프"))]
    start_key, start_date = parse_timestamp(timestamp_rows[0]["타임스탬프"]) if timestamp_rows else ((), "unknown")
    end_key, end_date = parse_timestamp(timestamp_rows[-1]["타임스탬프"]) if timestamp_rows else ((), "unknown")
    _ = (start_key, end_key)

    subjects_by_date: dict[str, set[str]] = defaultdict(set)
    rows_by_subject: dict[str, list[dict[str, str]]] = defaultdict(list)
    frequencies: dict[tuple[str, str, str], int] = defaultdict(int)

    for row in sorted_rows:
        subject = normalize(row.get("피험자"))
        domain = normalize(row.get("평가 도메인"))
        scenario = normalize(row.get("시나리오"))
        condition = condition_value(normalize(row.get("조건 선택")))
        _, date_label = parse_timestamp(row.get("타임스탬프", ""))

        if subject:
            subjects_by_date[date_label].add(subject)
            rows_by_subject[subject].append(row)
        if domain and scenario and condition:
            frequencies[(scenario, condition, domain)] += 1

    lines: list[str] = [
        "Experiment Metadata Summary",
        "===========================",
        "",
        f"N: {len(subjects)} participants",
        f"Response rows: {len(response_rows)}",
        f"Period: {start_date} to {end_date}",
        "",
        "Participants by date",
        "--------------------",
    ]

    for date_label in sorted(subjects_by_date):
        lines.append(f"{date_label}: {len(subjects_by_date[date_label])}")

    lines.extend(["", "Participant condition order", "---------------------------"])
    for subject in subjects:
        sequence_parts = []
        for row in sorted(rows_by_subject[subject], key=row_sort_key):
            domain = domain_sequence_code(normalize(row.get("평가 도메인")))
            scenario = scenario_code(normalize(row.get("시나리오")))
            condition = condition_code(normalize(row.get("조건 선택")))
            sequence_parts.append(f"{domain}_{scenario}{condition}")
        lines.append(f"{subject_code(subject)}: {'-'.join(sequence_parts)}")

    lines.extend(build_participant_accuracy_lines(subjects, sagat_rows))

    lines.extend(["", "S x C x Domain frequencies", "--------------------------"])
    scenarios = expected_scenarios(response_rows)
    domains = expected_domains(response_rows)
    lines.append(f"Combinations: {len(scenarios) * len(CONDITION_ORDER) * len(domains)}")
    lines.append(f"Total appearances: {sum(frequencies.values())}")
    lines.append("")
    for scenario in scenarios:
        for condition in CONDITION_ORDER:
            for domain in domains:
                count = frequencies[(scenario, condition, domain)]
                label = f"{scenario_code(scenario)}{condition_code(condition)}-{domain_code(domain)}"
                lines.append(f"{label}: {count}")

    return "\n".join(lines) + "\n"


def mean_and_sd(values: list[float]) -> tuple[float | None, float]:
    if not values:
        return None, 0.0
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, 0.0
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, math.sqrt(variance)


def build_output_rows(item_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    values_by_group: dict[tuple[str, str, str, str, str, str], list[float]] = defaultdict(list)
    for row in item_rows:
        for panel_label, domain_filter in DOMAIN_PANELS:
            if domain_filter is not None and row["평가 도메인"] != domain_filter:
                continue
            key = (
                row["척도"],
                row["문항"],
                row["문항내용"],
                panel_label,
                condition_value(row["조건 선택"]),
                "1" if row["척도"] == "SAGAT" else "7",
            )
            values_by_group[key].append(float(row["값"]))

    output_rows: list[dict[str, str]] = []
    for key in sorted(values_by_group, key=output_sort_key):
        scale, item_id, item_title, panel_label, condition, value_max = key
        values = values_by_group[key]
        mean, sd = mean_and_sd(values)
        output_rows.append(
            {
                "척도": scale,
                "문항": item_id,
                "문항내용": item_title,
                "도메인패널": panel_label,
                "조건 선택": condition,
                "평균": "" if mean is None else f"{mean:.6g}",
                "표준편차": f"{sd:.6g}",
                "표본수": str(len(values)),
                "값범위최소": "0",
                "값범위최대": value_max,
            }
        )

    return output_rows


def output_sort_key(key: tuple[str, str, str, str, str, str]) -> tuple[int, tuple[int, int], int, int]:
    scale, item_id, _, panel_label, condition, _ = key
    scale_index = 0 if scale == "SAGAT" else 1
    item_index = (0, int(item_id[1:])) if item_id.startswith("Q") else scale_question_sort_key(item_id)
    panel_index = next((index for index, panel in enumerate(DOMAIN_PANELS) if panel[0] == panel_label), 99)
    condition_index = next((index for index, value in enumerate(CONDITION_ORDER) if value == condition), 99)
    return scale_index, item_index, panel_index, condition_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Post-process response.csv into one condition-level output.csv file.",
    )
    parser.add_argument("--response", type=Path, default=DEFAULT_RESPONSE_PATH)
    parser.add_argument("--answer-sheet", type=Path, default=DEFAULT_ANSWER_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--metadata-output", type=Path, default=DEFAULT_METADATA_OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    response_rows = read_csv(args.response)
    answer_rows = read_csv(args.answer_sheet)
    answer_lookup = build_answer_lookup(answer_rows)

    sagat_rows = score_sagat_rows(response_rows, answer_lookup)
    item_rows = list(sagat_rows)
    item_rows.extend(read_scale_rows(response_rows))

    output_rows = build_output_rows(item_rows)
    write_csv(args.output, output_rows)
    metadata_summary = build_metadata_summary(response_rows, sagat_rows)
    write_text(args.metadata_output, metadata_summary)

    print(f"response: {args.response}")
    print(f"answer sheet: {args.answer_sheet}")
    print(f"output: {args.output}")
    print(f"metadata summary: {args.metadata_output}")
    print(f"rows: {len(output_rows)}")


if __name__ == "__main__":
    main()
