from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = BASE_DIR / "users_raw_data"
DEFAULT_OUTPUT_PATH = DEFAULT_INPUT_DIR / "피험자별결과.csv"

TIME_FILE = "BRL_논문_타임라인 정리.xlsx - 사용자평가-시간.csv"
SURVEY_FILE = "BRL_논문_타임라인 정리.xlsx - 피험자설문조사결과.csv"
SAGAT_ANSWER_FILE = "참고-SAGAT정답.csv"
SURVEY_TEXT_FILE = "text.text"
VIDEO_DURATION_FILE = "참고-도메인-시나리오별 동영상 시간.csv"

CONDITIONS = ("C1: All", "C2: No", "C3: Ours1", "C4: Ours2", "C5: KnowNo")
DOMAINS = ("Waste", "Tomato")
DOMAIN_KO_TO_EN = {
    "분리수거": "Waste",
    "토마토 수확": "Tomato",
}
DOMAIN_CODE_TO_EN = {
    "R": "Waste",
    "T": "Tomato",
}
CONDITION_CODE_TO_LABEL = {
    "1": "C1: All",
    "2": "C2: No",
    "3": "C3: Ours1",
    "4": "C4: Ours2",
    "5": "C5: KnowNo",
}
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
SCALE_MIN = 1.0
SCALE_MAX = 7.0
NASA_REVERSE_ITEMS = {"N4"}
SART_REVERSE_ITEMS = {"S4", "S5", "S7", "S8", "S9", "S10"}

BLOCK_HEADER_RE = re.compile(r"^([RT])([1-5])\s*\(\s*s([1-5])\s*\)")
ANSWER_TOKEN_RE = re.compile(r"\b([ABTF])\b", re.IGNORECASE)
SUBJECT_RE = re.compile(r"^p?0*(\d+)$", re.IGNORECASE)
NATURAL_DURATION_RE = re.compile(
    r"(?:(?P<hours>\d+)\s*(?:hours?|시간))?\s*"
    r"(?:(?P<minutes>\d+)\s*(?:minutes?|mins?|분))?\s*"
    r"(?:(?P<seconds>\d+)\s*(?:seconds?|secs?|초))?",
    re.IGNORECASE,
)

QUESTIONS = ("Q1", "Q2", "Q3", "Q4", "Q5", "Q6")
SAGAT_GROUPS = {
    "SAGAT1": ("Q1", "Q2"),
    "SAGAT2": ("Q3", "Q4"),
    "SAGAT3": ("Q5", "Q6"),
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
SURVEY_DOMAIN_LABELS = {
    "토마토 도메인": "토마토 수확",
    "분리수거 도메인": "분리수거",
}


def normalize(value: str | None) -> str:
    return " ".join((value or "").strip().split())


def read_rows(path: Path) -> list[list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.reader(file))


def read_dicts(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def cell(row: list[str], index: int) -> str:
    if 0 <= index < len(row):
        return row[index].strip()
    return ""


def subject_key(value: str) -> str:
    match = SUBJECT_RE.match(normalize(value))
    if match:
        return f"P{int(match.group(1)):03d}"
    return normalize(value)


def subject_sort_key(subject: str) -> tuple[int, str]:
    match = re.search(r"\d+", subject)
    return (int(match.group(0)) if match else 999999, subject)


def condition_label(value: str) -> str:
    text = normalize(value)
    match = re.search(r"C\s*([1-5])", text, re.IGNORECASE)
    if match:
        return CONDITION_CODE_TO_LABEL[match.group(1)]
    return text


def find_subject_above(rows: list[list[str]], header_row: int, col: int) -> str:
    for row_index in range(header_row - 1, -1, -1):
        for candidate_col in (col, col - 1, col + 1):
            value = cell(rows[row_index], candidate_col)
            if SUBJECT_RE.match(value):
                return subject_key(value)
    return ""


def split_answer_tokens(value: str) -> list[str]:
    tokens: list[str] = []
    for line in value.replace("\r", "\n").split("\n"):
        stripped = line.strip().upper()
        if stripped in {"N/A", "NA", "#N/A"}:
            continue
        found = ANSWER_TOKEN_RE.findall(stripped)
        tokens.extend(token.upper() for token in found)
    return tokens


def score_answer_cells(correct_cell: str, user_cell: str) -> tuple[int, int]:
    correct_tokens = split_answer_tokens(correct_cell)
    user_tokens = split_answer_tokens(user_cell)
    if not correct_tokens:
        return 0, 0

    correct = 0
    total = 0
    for index, expected in enumerate(correct_tokens):
        actual = user_tokens[index] if index < len(user_tokens) else None
        if actual is None:
            continue
        total += 1
        if actual == expected:
            correct += 1
    return correct, total


def parse_success_rates(input_dir: Path) -> dict[tuple[str, str, str], float]:
    successes: dict[tuple[str, str, str], list[int]] = defaultdict(list)

    for path in sorted(input_dir.glob("BRL_논문_타임라인 정리.xlsx - 6월*일.csv")):
        rows = read_rows(path)
        headers: list[tuple[int, int, str, str]] = []
        for row_index, row in enumerate(rows):
            for col_index, raw in enumerate(row):
                match = BLOCK_HEADER_RE.match(raw.strip())
                if not match:
                    continue
                if cell(row, col_index + 1) != "정답" or cell(row, col_index + 2) != "사용자답":
                    continue
                domain_code, condition_code, _scenario = match.groups()
                subject = find_subject_above(rows, row_index, col_index)
                if subject:
                    headers.append((row_index, col_index, domain_code, condition_code))

        header_rows = sorted({row_index for row_index, *_ in headers})
        for row_index, col_index, domain_code, condition_code in headers:
            subject = find_subject_above(rows, row_index, col_index)
            if not subject:
                continue
            end_row = next((candidate for candidate in header_rows if candidate > row_index), len(rows))
            key = (subject, DOMAIN_CODE_TO_EN[domain_code], CONDITION_CODE_TO_LABEL[condition_code])
            block_correct = 0
            block_total = 0

            for data_row in range(row_index + 1, end_row):
                correct, total = score_answer_cells(
                    cell(rows[data_row], col_index + 1),
                    cell(rows[data_row], col_index + 2),
                )
                block_correct += correct
                block_total += total

            if block_total:
                successes[key].append(1 if block_correct == block_total else 0)

    return {key: mean(values) * 100 for key, values in successes.items() if values}


def parse_duration_seconds(value: str) -> int | None:
    text = normalize(value)
    if not text:
        return None
    parts = text.split(":")
    if len(parts) in (2, 3):
        try:
            numbers = [int(part) for part in parts]
        except ValueError:
            return None
        if len(numbers) == 2:
            minutes, seconds = numbers
            return minutes * 60 + seconds
        hours, minutes, seconds = numbers
        return hours * 3600 + minutes * 60 + seconds

    match = NATURAL_DURATION_RE.fullmatch(text)
    if not match or not any(match.groupdict().values()):
        return None
    hours = int(match.group("hours") or 0)
    minutes = int(match.group("minutes") or 0)
    seconds = int(match.group("seconds") or 0)
    return hours * 3600 + minutes * 60 + seconds


def load_video_durations(path: Path) -> dict[tuple[str, str], int]:
    durations: dict[tuple[str, str], int] = {}
    for row in read_dicts(path):
        domain_code = normalize(row.get("domain_code"))
        scenario = normalize(row.get("scenario"))
        if not domain_code or not scenario:
            continue

        seconds = parse_duration_seconds(row.get("video_seconds", ""))
        if seconds is None:
            seconds = parse_duration_seconds(row.get("video_hms", ""))
        if seconds is not None:
            durations[(domain_code, scenario)] = seconds
    return durations


def parse_operation_times(path: Path, video_durations: dict[tuple[str, str], int]) -> dict[tuple[str, str, str], float]:
    rows = read_rows(path)
    values: dict[tuple[str, str, str], list[int]] = defaultdict(list)

    for row_index, row in enumerate(rows):
        for col_index, raw in enumerate(row):
            match = BLOCK_HEADER_RE.match(raw.strip())
            if not match:
                continue
            domain_code, condition_code, scenario = match.groups()
            subject = find_subject_above(rows, row_index, col_index)
            if not subject:
                continue

            for offset in range(1, 10):
                data_row = row_index + offset
                if data_row >= len(rows):
                    break
                if cell(rows[data_row], col_index + 1) != "총 조작 시간":
                    continue
                seconds = parse_duration_seconds(cell(rows[data_row], col_index + 4))
                if seconds is not None:
                    seconds = max(0, seconds - video_durations.get((domain_code, scenario), 0))
                    key = (subject, DOMAIN_CODE_TO_EN[domain_code], CONDITION_CODE_TO_LABEL[condition_code])
                    values[key].append(seconds)
                break

    return {key: mean(seconds) for key, seconds in values.items() if seconds}


def read_survey_option_codes(path: Path) -> dict[str, dict[str, dict[str, str]]]:
    codes: dict[str, dict[str, dict[str, str]]] = {domain: {} for domain in DOMAIN_KO_TO_EN}
    current_domain: str | None = None
    current_question: str | None = None
    option_index = 1

    for line in path.read_text(encoding="utf-8").splitlines():
        text = normalize(line)
        if not text:
            continue
        if text.startswith("<") and text.endswith(">") and text != "<SAGAT>":
            break
        if text in SURVEY_DOMAIN_LABELS:
            current_domain = SURVEY_DOMAIN_LABELS[text]
            current_question = None
            option_index = 1
            continue
        question_match = re.match(r"^(Q[1-6])\.", text)
        if question_match and current_domain is not None:
            current_question = question_match.group(1)
            codes[current_domain][current_question] = {}
            option_index = 1
            continue
        if current_domain and current_question:
            codes[current_domain][current_question][text] = str(option_index)
            option_index += 1

    return codes


def unique_header_indices(header: list[str]) -> dict[str, int]:
    indices: dict[str, int] = {}
    for index, column in enumerate(header):
        indices.setdefault(normalize(column), index)
    return indices


def find_question_columns(header: list[str]) -> dict[str, dict[str, int]]:
    normalized_header = [normalize(column) for column in header]
    columns: dict[str, dict[str, int]] = {domain: {} for domain in DOMAIN_KO_TO_EN}
    used: set[int] = set()

    for domain, prefixes in DOMAIN_QUESTION_PREFIXES.items():
        for question, prefix in prefixes.items():
            for index, column in enumerate(normalized_header):
                if index in used:
                    continue
                if column.startswith(prefix):
                    columns[domain][question] = index
                    used.add(index)
                    break
    return columns


def code_sagat_response(
    option_codes: dict[str, dict[str, dict[str, str]]],
    domain: str,
    question: str,
    value: str,
) -> str:
    text = normalize(value)
    if not text:
        return ""
    if text.isdigit():
        return text
    return option_codes.get(domain, {}).get(question, {}).get(text, text)


def scale_item_id(column: str) -> str | None:
    item = normalize(column).split(".", 1)[0]
    if item == "F":
        return item
    if len(item) > 1 and item[0] in {"N", "S"} and item[1:].isdigit():
        return item
    return None


def parse_float(value: str) -> float | None:
    text = normalize(value)
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def lower_is_better_value(item_id: str, value: float) -> float:
    if item_id in NASA_REVERSE_ITEMS or item_id in SART_REVERSE_ITEMS:
        return SCALE_MIN + SCALE_MAX - value
    return value


def parse_survey_metrics(input_dir: Path) -> dict[tuple[str, str, str, str], float]:
    survey_path = input_dir / SURVEY_FILE
    answer_path = input_dir / SAGAT_ANSWER_FILE
    option_path = input_dir / SURVEY_TEXT_FILE

    rows = read_rows(survey_path)
    if not rows:
        return {}
    header, data_rows = rows[0], rows[1:]
    indices = unique_header_indices(header)
    question_columns = find_question_columns(header)
    option_codes = read_survey_option_codes(option_path)
    answer_lookup = {
        (
            normalize(row["평가 도메인"]),
            normalize(row["시나리오"]),
            condition_label(row["조건 선택"]),
        ): {question: normalize(row[question]) for question in QUESTIONS}
        for row in read_dicts(answer_path)
    }
    scale_columns = {
        item_id: index
        for index, column in enumerate(header)
        if (item_id := scale_item_id(column)) is not None
    }

    metrics: dict[tuple[str, str, str, str], float] = {}
    for row in data_rows:
        subject = subject_key(cell(row, indices.get("피험자", -1)))
        domain_ko = normalize(cell(row, indices.get("평가 도메인", -1)))
        scenario = normalize(cell(row, indices.get("시나리오", -1)))
        condition = condition_label(cell(row, indices.get("조건 선택", -1)))
        domain = DOMAIN_KO_TO_EN.get(domain_ko, "")
        if not subject or not domain:
            continue

        sagat_scores: dict[str, float] = {}
        answers = answer_lookup.get((domain_ko, scenario, condition), {})
        for question in QUESTIONS:
            column_index = question_columns.get(domain_ko, {}).get(question)
            if column_index is None:
                continue
            response = code_sagat_response(option_codes, domain_ko, question, cell(row, column_index))
            answer = answers.get(question, "")
            if response and answer:
                sagat_scores[question] = 1.0 if response == answer else 0.0

        for metric, questions in SAGAT_GROUPS.items():
            values = [sagat_scores[question] for question in questions if question in sagat_scores]
            if values:
                metrics[(subject, domain, condition, metric)] = mean(values)

        fatigue = parse_float(cell(row, scale_columns.get("F", -1)))
        if fatigue is not None:
            metrics[(subject, domain, condition, "Fatigue")] = fatigue

        nasa_values = []
        for item in ("N1", "N2", "N3", "N4", "N5", "N6"):
            value = parse_float(cell(row, scale_columns.get(item, -1)))
            if value is not None:
                nasa_values.append(lower_is_better_value(item, value))
        if nasa_values:
            metrics[(subject, domain, condition, "NASA-RTLX")] = mean(nasa_values)

        sart_values = []
        for item in ("S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10"):
            value = parse_float(cell(row, scale_columns.get(item, -1)))
            if value is not None:
                sart_values.append(lower_is_better_value(item, value))
        if sart_values:
            metrics[(subject, domain, condition, "SART")] = mean(sart_values)

    return metrics


def format_value(value: float | None, metric: str) -> str:
    if value is None:
        return ""
    if metric in {"조작성공률 (%)", "조작시간 (s)"}:
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return f"{value:.6g}"


def write_output(
    output_path: Path,
    subjects: list[str],
    values: dict[tuple[str, str, str, str], float],
    subjects_per_block: int,
) -> None:
    rows: list[list[str]] = []
    for start in range(0, len(subjects), subjects_per_block):
        block_subjects = subjects[start : start + subjects_per_block]
        header_subjects = ["조건", "도메인"]
        header_metrics = ["", ""]
        for subject in block_subjects:
            header_subjects.extend([subject] + [""] * (len(METRICS) - 1))
            header_metrics.extend(METRICS)
        rows.append(header_subjects)
        rows.append(header_metrics)

        for condition in CONDITIONS:
            for domain_index, domain in enumerate(DOMAINS):
                row = [condition if domain_index == 0 else "", domain]
                for subject in block_subjects:
                    for metric in METRICS:
                        row.append(format_value(values.get((subject, domain, condition, metric)), metric))
                rows.append(row)
        rows.append([])
        rows.append([])

    rows.append(["조건", "도메인", "P_ALL"] + [""] * (len(METRICS) - 1))
    rows.append(["", "", *METRICS])
    for condition in CONDITIONS:
        for domain_index, domain in enumerate(DOMAINS):
            row = [condition if domain_index == 0 else "", domain]
            for metric in METRICS:
                metric_values = [
                    values[(subject, domain, condition, metric)]
                    for subject in subjects
                    if (subject, domain, condition, metric) in values
                ]
                row.append(format_value(mean(metric_values), metric) if metric_values else "")
            rows.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(rows)


def build_results(input_dir: Path) -> tuple[dict[tuple[str, str, str, str], float], list[str]]:
    values: dict[tuple[str, str, str, str], float] = {}
    video_durations = load_video_durations(input_dir / VIDEO_DURATION_FILE)

    for key, value in parse_success_rates(input_dir).items():
        values[(*key, "조작성공률 (%)")] = value
    for key, value in parse_operation_times(input_dir / TIME_FILE, video_durations).items():
        values[(*key, "조작시간 (s)")] = value
    values.update(parse_survey_metrics(input_dir))

    subjects = sorted({subject for subject, *_ in values}, key=subject_sort_key)
    return values, subjects


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="피험자별 결과 CSV를 생성합니다.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--subjects-per-block", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    values, subjects = build_results(args.input_dir)
    write_output(args.output, subjects, values, args.subjects_per_block)

    print(f"subjects: {len(subjects)}")
    print(f"values: {len(values)}")
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()
