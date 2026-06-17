from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
SURVEY_DATA_DIR = BASE_DIR / "data" / "03_user_survey"
DEFAULT_RESPONSE_PATH = SURVEY_DATA_DIR / "response.csv"
DEFAULT_ANSWER_PATH = SURVEY_DATA_DIR / "answer_sheet.csv"
DEFAULT_SURVEY_PATH = SURVEY_DATA_DIR / "survey.txt"
DEFAULT_OUTPUT_PATH = SURVEY_DATA_DIR / "output.csv"
DEFAULT_METADATA_OUTPUT_PATH = SURVEY_DATA_DIR / "metadata_summary.txt"

KEY_FIELDS = ("평가 도메인", "시나리오", "조건 선택")
QUESTIONS = ("Q1", "Q2", "Q3", "Q4", "Q5", "Q6")
SCENARIO_ORDER = ("시나리오1", "시나리오2", "시나리오3", "시나리오4", "시나리오5")
DOMAIN_ORDER = ("토마토 수확", "분리수거")
DOMAIN_PANELS: tuple[tuple[str, str | None], ...] = (
    ("토마토 수확", "토마토 수확"),
    ("분리수거", "분리수거"),
    ("통합", None),
)
CONDITION_ORDER = ("C1: All", "C2: No", "C3: Ours1", "C4: Ours2", "C5: KnowNo")
SCALE_MIN = 1.0
SCALE_MAX = 7.0
REVERSE_CODED_ITEMS: tuple[str, ...] = ()

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

SAGAT_ITEM_TITLES = {
    "Q1": "Q1. 현재 화면의 대상 물체 개수",
    "Q2": "Q2. 로봇이 마지막으로 처리한 대상의 상태/재질",
    "Q3": "Q3. 현재 상황에서 사용자가 판단해야 하는 핵심 정보",
    "Q4": "Q4. 현재 상황에서 사용자의 확인이 필요한 이유",
    "Q5": "Q5. 사용자가 아무 행동도 하지 않을 때 예상되는 로봇 행동",
    "Q6": "Q6. 잘못된 로봇 작업을 수정하지 않을 때 예상되는 결과",
}

DOMAIN_QUESTION_PREFIXES = {
    "토마토 수확": {
        "Q1": "Q1. 현재 화면에 표시된 토마토는",
        "Q2": "Q2. 로봇이 마지막으로 처리한 토마토의 상태",
        "Q3": "Q3. 현재 상황에서 사용자가 판단해야 하는 핵심 정보",
        "Q4": "Q4. 현재 상황에서 사용자의 확인이 필요한 이유",
        "Q5": "Q5. 현재 사용자가 아무 행동도 하지 않으면",
        "Q6": "Q6. 로봇이 잘못된 토마토를 선택했는데",
    },
    "분리수거": {
        "Q1": "Q1. 현재 화면에서 테이블 위에 놓여있는 물건",
        "Q2": "Q2. 로봇이 마지막으로 처리한 쓰레기의 재질",
        "Q3": "Q3. 현재 상황에서 사용자가 판단해야 하는 핵심 정보",
        "Q4": "Q4. 방금 상황에서 사용자의 확인이 필요한 이유",
        "Q5": "Q5. 현재 사용자가 아무 행동도 하지 않으면",
        "Q6": "Q6. 로봇이 잘못된 분리수거를 했는데",
    },
}

FALLBACK_OPTION_CODES = {
    "토마토 수확": {
        "Q1": {"0개": "1", "1개": "2", "2개": "3", "잘 모르겠다": "4"},
        "Q2": {"익음": "1", "덜 익음": "2", "손상됨": "3", "잘 모르겠다": "4"},
        "Q3": {
            "목표한 토마토가 수확 가능한지 여부": "1",
            "로봇이 다음 위치로 이동해야 하는지 여부": "2",
            "로봇이 잘못 판단한 정보가 있는지 여부": "3",
            "잘 모르겠다": "4",
        },
        "Q4": {
            "토마토의 성숙도 판단이 불확실하기 때문": "1",
            "토마토의 손상 여부 판단이 불확실하기 때문": "2",
            "수확 대상에서 빠진 토마토가 있기 때문": "3",
            "잘 모르겠다": "4",
        },
        "Q5": {"대기 상태를 유지함": "1", "현재 판단대로 작업을 계속 진행함": "2", "작업을 종료함": "3", "잘 모르겠다": "4"},
        "Q6": {
            "잘못된 토마토를 수확할 수 있음": "1",
            "로봇이 자동으로 올바른 토마토를 다시 선택함": "2",
            "작업이 즉시 종료됨": "3",
            "잘 모르겠다": "4",
        },
    },
    "분리수거": {
        "Q1": {"0개": "1", "1개": "2", "2개 이상": "3", "잘 모르겠다": "4"},
        "Q2": {"플라스틱": "1", "캔": "2", "일반쓰레기": "3", "종이": "4"},
        "Q3": {
            "다음 물체의 존재유무": "1",
            "대상 물체가 현재 위치에서 잡을 수 있는지 여부": "2",
            "대상 물체가 올바른 분리수거 통으로 이동했는지 여부": "3",
            "잘 모르겠다": "4",
        },
        "Q4": {
            "대상 물체의 분리수거 품목 판단이 불확실하기 때문": "1",
            "대상 물체가 로봇이 잡을 수 있는 범위 밖에 있기 때문": "2",
            "대상 물체가 오염되어 분리수거 가능 여부를 판단해야 하기 때문": "3",
            "잘 모르겠다": "4",
        },
        "Q5": {"대기 상태를 유지함": "1", "현재 판단대로 분리수거 작업을 계속 진행함": "2", "작업을 종료함": "3", "잘 모르겠다": "4"},
        "Q6": {
            "대상 물체가 잘못된 분류함에 들어갈 수 있음": "1",
            "로봇이 자동으로 오류를 인식하고 다시 분류함": "2",
            "로봇이 모든 분리수거 작업을 즉시 종료함": "3",
            "잘 모르겠다": "4",
        },
    },
}

SURVEY_DOMAIN_LABELS = {
    "토마토 도메인": "토마토 수확",
    "분리수거 도메인": "분리수거",
}


def normalize(value: str | None) -> str:
    return " ".join((value or "").strip().split())


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def read_csv_table(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file)
        rows = list(reader)
    if not rows:
        return [], []
    return rows[0], rows[1:]


def read_survey_option_codes(path: Path) -> dict[str, dict[str, dict[str, str]]]:
    if not path.exists():
        return FALLBACK_OPTION_CODES

    lines = [normalize(line) for line in path.read_text(encoding="utf-8").splitlines()]
    codes: dict[str, dict[str, dict[str, str]]] = {domain: {} for domain in DOMAIN_ORDER}
    current_domain: str | None = None
    current_question: str | None = None
    option_index = 1

    for line in lines:
        if not line:
            continue
        if line.startswith("<") and line.endswith(">") and line != "<SAGAT>":
            break
        if line in SURVEY_DOMAIN_LABELS:
            current_domain = SURVEY_DOMAIN_LABELS[line]
            current_question = None
            option_index = 1
            continue
        question_match = re.match(r"^(Q[1-6])\.", line)
        if question_match and current_domain is not None:
            current_question = question_match.group(1)
            codes[current_domain][current_question] = {}
            option_index = 1
            continue
        if current_domain is not None and current_question is not None:
            codes[current_domain][current_question][line] = str(option_index)
            option_index += 1

    for domain in DOMAIN_ORDER:
        for question in QUESTIONS:
            if not codes.get(domain, {}).get(question):
                return FALLBACK_OPTION_CODES
    return codes


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def row_get(row: list[str], index: int | None) -> str:
    if index is None or index >= len(row):
        return ""
    return row[index]


def unique_header_indices(header: list[str]) -> dict[str, int]:
    indices: dict[str, int] = {}
    for index, column in enumerate(header):
        indices.setdefault(normalize(column), index)
    return indices


def find_question_columns(header: list[str]) -> dict[str, dict[str, int]]:
    normalized_header = [normalize(column) for column in header]
    columns: dict[str, dict[str, int]] = {domain: {} for domain in DOMAIN_ORDER}
    used: set[int] = set()

    for domain in DOMAIN_ORDER:
        for question in QUESTIONS:
            prefix = DOMAIN_QUESTION_PREFIXES[domain][question]
            matches = [
                index
                for index, column in enumerate(normalized_header)
                if index not in used and column.startswith(prefix)
            ]
            if not matches:
                continue
            selected = matches[0]
            columns[domain][question] = selected
            used.add(selected)

    return columns


def row_key(row: dict[str, str]) -> tuple[str, str, str]:
    return tuple(normalize(row.get(field)) for field in KEY_FIELDS)


def build_answer_lookup(answer_rows: list[dict[str, str]]) -> dict[tuple[str, str, str], dict[str, str]]:
    lookup: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in answer_rows:
        lookup[row_key(row)] = {question: normalize(row.get(question)) for question in QUESTIONS}
    return lookup


def condition_value(condition: str) -> str:
    for expected in CONDITION_ORDER:
        code = expected.split(":", 1)[0]
        if condition == expected or condition.startswith(code):
            return expected
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


def code_response(
    option_codes: dict[str, dict[str, dict[str, str]]],
    domain: str,
    question: str,
    value: str,
) -> str:
    normalized = normalize(value)
    if not normalized:
        return ""
    if normalized.isdigit():
        return normalized
    return option_codes.get(domain, {}).get(question, {}).get(normalized, normalized)


def score_sagat_rows(
    header: list[str],
    response_rows: list[list[str]],
    answer_lookup: dict[tuple[str, str, str], dict[str, str]],
    option_codes: dict[str, dict[str, dict[str, str]]],
) -> tuple[list[dict[str, str]], list[str]]:
    item_rows: list[dict[str, str]] = []
    warnings: list[str] = []
    indices = unique_header_indices(header)
    q_columns = find_question_columns(header)

    for domain in DOMAIN_ORDER:
        missing = [question for question in QUESTIONS if question not in q_columns.get(domain, {})]
        if missing:
            warnings.append(f"Missing response columns for {domain}: {', '.join(missing)}")

    for row_index, row in enumerate(response_rows, start=2):
        subject = normalize(row_get(row, indices.get("피험자")))
        domain = normalize(row_get(row, indices.get("평가 도메인")))
        scenario = normalize(row_get(row, indices.get("시나리오")))
        condition = condition_value(normalize(row_get(row, indices.get("조건 선택"))))
        answers = answer_lookup.get((domain, scenario, condition))
        if answers is None:
            warnings.append(f"Missing answer key at row {row_index}: {domain}, {scenario}, {condition}")
            continue

        for question in QUESTIONS:
            column_index = q_columns.get(domain, {}).get(question)
            response_value = code_response(option_codes, domain, question, row_get(row, column_index))
            answer_value = normalize(answers.get(question))
            if not response_value or not answer_value:
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

    return item_rows, warnings


def item_id_from_scale_column(column: str) -> str | None:
    item_id = normalize(column).split(".", 1)[0]
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


def response_sort_key(row: list[str], indices: dict[str, int]) -> tuple[int, int, int, int, int, int, int]:
    return parse_timestamp(row_get(row, indices.get("타임스탬프")))[0]


def format_rate(correct: int, total: int) -> str:
    if total == 0:
        return "NA"
    return f"{correct / total:.3f}"


def expected_scenarios(header: list[str], response_rows: list[list[str]]) -> list[str]:
    indices = unique_header_indices(header)
    scenarios = {
        normalize(row_get(row, indices.get("시나리오")))
        for row in response_rows
        if normalize(row_get(row, indices.get("시나리오")))
    }
    ordered = list(SCENARIO_ORDER)
    ordered.extend(
        sorted(
            scenarios - set(ordered),
            key=lambda value: int(re.search(r"\d+", value).group(0)) if re.search(r"\d+", value) else 999,
        )
    )
    return ordered


def expected_domains(header: list[str], response_rows: list[list[str]]) -> list[str]:
    indices = unique_header_indices(header)
    observed = {
        normalize(row_get(row, indices.get("평가 도메인")))
        for row in response_rows
        if normalize(row_get(row, indices.get("평가 도메인")))
    }
    ordered = list(DOMAIN_ORDER)
    ordered.extend(sorted(observed - set(ordered)))
    return ordered


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


def read_scale_rows(header: list[str], response_rows: list[list[str]]) -> list[dict[str, str]]:
    scale_columns = {
        item_id: index
        for index, column in enumerate(header)
        if (item_id := item_id_from_scale_column(column)) is not None
    }
    scale_columns = dict(sorted(scale_columns.items(), key=lambda item: scale_question_sort_key(item[0])))
    indices = unique_header_indices(header)
    item_rows: list[dict[str, str]] = []

    for row in response_rows:
        domain = normalize(row_get(row, indices.get("평가 도메인")))
        condition = condition_value(normalize(row_get(row, indices.get("조건 선택"))))
        for item_id, column_index in scale_columns.items():
            value = coded_scale_value(item_id, parse_float(row_get(row, column_index)))
            if value is None:
                continue
            item_rows.append(
                {
                    "척도": "Survey",
                    "문항": item_id,
                    "문항내용": normalize(header[column_index]),
                    "평가 도메인": domain,
                    "조건 선택": condition,
                    "값": f"{value:g}",
                }
            )

    return item_rows


def mean_and_sd(values: list[float]) -> tuple[float | None, float]:
    if not values:
        return None, 0.0
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, 0.0
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, math.sqrt(variance)


def output_sort_key(key: tuple[str, str, str, str, str, str]) -> tuple[int, tuple[int, int], int, int]:
    scale, item_id, _, panel_label, condition, _ = key
    scale_index = 0 if scale == "SAGAT" else 1
    if item_id.startswith("Q"):
        item_index = (0, int(item_id[1:]))
    else:
        item_index = scale_question_sort_key(item_id)
    panel_index = next((index for index, panel in enumerate(DOMAIN_PANELS) if panel[0] == panel_label), 99)
    condition_index = next((index for index, value in enumerate(CONDITION_ORDER) if value == condition), 99)
    return scale_index, item_index, panel_index, condition_index


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


def subject_sort_key(subject: str) -> tuple[int, str]:
    match = re.search(r"\d+", subject)
    return (int(match.group(0)) if match else 999999, subject)


def build_participant_accuracy_lines(subjects: list[str], sagat_rows: list[dict[str, str]]) -> list[str]:
    rows_by_subject: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in sagat_rows:
        rows_by_subject[row.get("피험자", "")].append(row)

    lines = ["", "Participant SAGAT accuracy", "--------------------------"]
    for subject in subjects:
        subject_rows = rows_by_subject.get(subject, [])
        total_correct = 0
        total_scored = 0
        parts = []
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


def build_metadata_summary(
    header: list[str],
    response_rows: list[list[str]],
    sagat_rows: list[dict[str, str]],
    warnings: list[str],
    option_codes: dict[str, dict[str, dict[str, str]]],
) -> str:
    indices = unique_header_indices(header)
    subjects = sorted(
        {normalize(row_get(row, indices.get("피험자"))) for row in response_rows if normalize(row_get(row, indices.get("피험자")))},
        key=subject_sort_key,
    )
    sorted_rows = sorted(response_rows, key=lambda row: response_sort_key(row, indices))
    timestamp_rows = [row for row in sorted_rows if normalize(row_get(row, indices.get("타임스탬프")))]
    if timestamp_rows:
        _, start_date = parse_timestamp(row_get(timestamp_rows[0], indices.get("타임스탬프")))
        _, end_date = parse_timestamp(row_get(timestamp_rows[-1], indices.get("타임스탬프")))
    else:
        start_date = "unknown"
        end_date = "unknown"

    subjects_by_date: dict[str, set[str]] = defaultdict(set)
    rows_by_subject: dict[str, list[list[str]]] = defaultdict(list)
    frequencies: dict[tuple[str, str, str], int] = defaultdict(int)

    for row in sorted_rows:
        subject = normalize(row_get(row, indices.get("피험자")))
        domain = normalize(row_get(row, indices.get("평가 도메인")))
        scenario = normalize(row_get(row, indices.get("시나리오")))
        condition = condition_value(normalize(row_get(row, indices.get("조건 선택"))))
        _, date_label = parse_timestamp(row_get(row, indices.get("타임스탬프")))

        if subject:
            subjects_by_date[date_label].add(subject)
            rows_by_subject[subject].append(row)
        if domain and scenario and condition:
            frequencies[(scenario, condition, domain)] += 1

    lines = [
        "Experiment Metadata Summary",
        "===========================",
        "",
        f"N: {len(subjects)} participants",
        f"Response rows: {len(response_rows)}",
        f"Period: {start_date} to {end_date}",
        f"SAGAT scored rows: {len(sagat_rows)}",
        "",
        "Participants by date",
        "--------------------",
    ]

    for date_label in sorted(subjects_by_date):
        lines.append(f"{date_label}: {len(subjects_by_date[date_label])}")

    lines.extend(["", "Participant condition order", "---------------------------"])
    for subject in subjects:
        sequence_parts = []
        for row in sorted(rows_by_subject[subject], key=lambda item: response_sort_key(item, indices)):
            domain = domain_sequence_code(normalize(row_get(row, indices.get("평가 도메인"))))
            scenario = scenario_code(normalize(row_get(row, indices.get("시나리오"))))
            condition = condition_code(condition_value(normalize(row_get(row, indices.get("조건 선택")))))
            sequence_parts.append(f"{domain}_{scenario}{condition}")
        lines.append(f"{subject_code(subject)}: {'-'.join(sequence_parts)}")

    lines.extend(build_participant_accuracy_lines(subjects, sagat_rows))

    lines.extend(["", "S x C x Domain frequencies", "--------------------------"])
    scenarios = expected_scenarios(header, response_rows)
    domains = expected_domains(header, response_rows)
    lines.append(f"Combinations: {len(scenarios) * len(CONDITION_ORDER) * len(domains)}")
    lines.append(f"Total appearances: {sum(frequencies.values())}")
    lines.append("")
    for scenario in scenarios:
        for condition in CONDITION_ORDER:
            for domain in domains:
                count = frequencies[(scenario, condition, domain)]
                label = f"{scenario_code(scenario)}{condition_code(condition)}-{domain_code(domain)}"
                lines.append(f"{label}: {count}")

    if warnings:
        lines.extend(["", "Warnings", "--------"])
        lines.extend(warnings)

    lines.extend(["", "SAGAT option coding", "-------------------"])
    for domain in DOMAIN_ORDER:
        lines.append(domain)
        for question in QUESTIONS:
            options = option_codes.get(domain, {}).get(question, {})
            option_text = ", ".join(f"{code}={text}" for text, code in options.items())
            lines.append(f"{question}: {option_text}")

    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post-process user-study responses into output.csv.")
    parser.add_argument("--response", type=Path, default=DEFAULT_RESPONSE_PATH)
    parser.add_argument("--answer-sheet", type=Path, default=DEFAULT_ANSWER_PATH)
    parser.add_argument("--survey", type=Path, default=DEFAULT_SURVEY_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--metadata-output", type=Path, default=DEFAULT_METADATA_OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    header, response_rows = read_csv_table(args.response)
    answer_rows = read_csv_dicts(args.answer_sheet)
    answer_lookup = build_answer_lookup(answer_rows)
    option_codes = read_survey_option_codes(args.survey)

    sagat_rows, warnings = score_sagat_rows(header, response_rows, answer_lookup, option_codes)
    item_rows = list(sagat_rows)
    item_rows.extend(read_scale_rows(header, response_rows))

    output_rows = build_output_rows(item_rows)
    write_csv(args.output, output_rows)
    write_text(args.metadata_output, build_metadata_summary(header, response_rows, sagat_rows, warnings, option_codes))

    print(f"response: {args.response}")
    print(f"answer sheet: {args.answer_sheet}")
    print(f"survey: {args.survey}")
    print(f"output: {args.output}")
    print(f"metadata summary: {args.metadata_output}")
    print(f"rows: {len(output_rows)}")
    print(f"sagat scored rows: {len(sagat_rows)}")
    if warnings:
        print(f"warnings: {len(warnings)}")


if __name__ == "__main__":
    main()
