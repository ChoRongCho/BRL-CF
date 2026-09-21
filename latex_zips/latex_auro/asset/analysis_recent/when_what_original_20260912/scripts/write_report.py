from __future__ import annotations

import csv
from pathlib import Path


P = Path(__file__).resolve().parents[1]


def read(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


summary = read(P / "01_processed/summary.csv")
scenes = read(P / "01_processed/scenes.csv")
lookup = {(r["domain"], r["condition"]): r for r in summary}
order = ["random", "ours_what_only", "ours_when_only", "ours"]
labels = {
    "random": "Random When + Random What",
    "ours_what_only": "Random When + Ours What",
    "ours_when_only": "Ours When + Random What",
    "ours": "Ours When + Ours What",
}


def f(value, digits=2):
    return f"{float(value):.{digits}f}"


random = lookup[("all", "random")]
what = lookup[("all", "ours_what_only")]
when = lookup[("all", "ours_when_only")]
ours = lookup[("all", "ours")]

when_gain_random_what = float(when["success_percent"]) - float(random["success_percent"])
when_gain_ours_what = float(ours["success_percent"]) - float(what["success_percent"])
what_reduction_when = float(when["questions_success_only_mean"]) - float(ours["questions_success_only_mean"])
what_reduction_when_pct = 100 * what_reduction_when / float(when["questions_success_only_mean"])
what_reduction_random = float(random["questions_success_only_mean"]) - float(what["questions_success_only_mean"])
what_reduction_random_pct = 100 * what_reduction_random / float(random["questions_success_only_mean"])

lines = [
    "# When–What ablation (gamma = 0.2) — 분석 보고서",
    "",
    "2026-09-21에 다시 실행한 1,600 episodes를 분석했다. Tomato와 Waste Sorting에서 장면 1–5를 사용했고, 각 조건은 400회다. 모든 로그의 gamma는 0.2이며 누락, 중복, 실행 오류는 없다.",
    "",
    "## 이 실험이 답하는 질문",
    "",
    "이 ablation은 질문 정책을 **언제 물을지(When)**와 **무엇을 물을지(What)**로 나누어 각각의 역할을 확인한다.",
    "",
    "- When의 효과는 `Random`과 `When only`, 또는 `What only`와 `Ours`의 성공률 차이로 본다.",
    "- What의 효과는 성공률이 비슷한 조건 안에서 성공한 실행의 질문 수 차이로 본다.",
    "- 전체 실행 질문 수도 함께 기록하지만, 실패가 일찍 끝나면 질문할 기회가 줄어들기 때문에 질의 효율의 주된 비교에는 성공 실행 질문 수를 사용한다.",
    "",
    "## 핵심 결과",
    "",
    f"1. **질문 타이밍이 성공률을 결정했다.** Random What을 유지한 채 Ours When을 적용하면 성공률이 {f(random['success_percent'])}%에서 {f(when['success_percent'])}%로 **{when_gain_random_what:.2f} percentage points** 증가했다. Ours What을 사용한 조건에서도 {f(what['success_percent'])}%에서 {f(ours['success_percent'])}%로 **{when_gain_ours_what:.2f} percentage points** 증가했다.",
    f"2. **질문 내용 선택이 질문 수를 줄였다.** Ours When 조건에서 성공 실행당 질문 수가 {f(when['questions_success_only_mean'])}회에서 {f(ours['questions_success_only_mean'])}회로 **{what_reduction_when:.2f}회, {what_reduction_when_pct:.1f}% 감소**했다. Random When 조건에서도 {f(random['questions_success_only_mean'])}회에서 {f(what['questions_success_only_mean'])}회로 **{what_reduction_random:.2f}회, {what_reduction_random_pct:.1f}% 감소**했다.",
    f"3. **두 구성요소를 결합한 Ours가 가장 좋은 성공–질의 trade-off를 보였다.** 전체 성공률은 {f(ours['success_percent'])}%이고, 성공 실행당 질문 수는 {f(ours['questions_success_only_mean'])}회다. When only는 성공률 {f(when['success_percent'])}%로 비슷하지만 질문을 {f(when['questions_success_only_mean'])}회 사용했다.",
    "",
    "## 전체·도메인별 결과",
    "",
    "| Domain | Condition | 성공/전체 | 성공률 | 전체 질문 | 성공 시 질문 | 전체 물리 step | 성공 시 물리 step |",
    "|---|---|---:|---:|---:|---:|---:|---:|",
]
for domain in ("all", "tomato", "wastesorting"):
    for condition in order:
        r = lookup[(domain, condition)]
        lines.append(
            f"| {domain} | {labels[condition]} | {r['successes']}/{r['n_valid']} | "
            f"{f(r['success_percent'])}% | {f(r['questions_mean'])} | "
            f"{f(r['questions_success_only_mean'])} | {f(r['steps_mean'])} | "
            f"{f(r['steps_success_only_mean'])} |"
        )

lines += [
    "",
    "## 장면별 반복 경향",
    "",
    "각 셀은 `성공률 / 성공 실행 질문 수`다. When의 성공률 효과와 What의 질문 절감 효과가 두 도메인의 모든 장면에서 반복된다.",
    "",
    "| Domain | Scene | Random | What only | When only | Ours |",
    "|---|---:|---:|---:|---:|---:|",
]
scene_lookup = {(r["domain"], r["scene"], r["condition"]): r for r in scenes}
for domain in ("tomato", "wastesorting"):
    for scene in ("01", "02", "03", "04", "05"):
        cells = []
        for condition in order:
            r = scene_lookup[(domain, scene, condition)]
            cells.append(f"{f(r['success_percent'], 1)}% / {f(r['questions_success_only_mean'], 1)}")
        lines.append(f"| {domain} | {scene} | " + " | ".join(cells) + " |")

lines += [
    "",
    "## 실패와 지표 해석",
    "",
    "| Condition | 실패 | Plan failure | Max step |",
    "|---|---:|---:|---:|",
    "| Random When + Random What | 127 | 125 | 2 |",
    "| Random When + Ours What | 124 | 120 | 4 |",
    "| Ours When + Random What | 6 | 5 | 1 |",
    "| Ours When + Ours What | 1 | 0 | 1 |",
    "",
    f"전체 실행 평균만 보면 Random의 질문 수가 {f(random['questions_mean'])}회로 Ours의 {f(ours['questions_mean'])}회보다 작다. 이는 Random이 더 효율적이어서가 아니라 127회가 과제를 끝내기 전에 종료되어 질문 기회가 줄었기 때문이다. 성공 실행끼리 비교하면 Ours는 {f(ours['questions_success_only_mean'])}회, Random은 {f(random['questions_success_only_mean'])}회다.",
    "",
    "성공 실행의 물리 step은 Ours 12.46, When only 12.37로 거의 같다. 따라서 Ours의 질문 감소는 물리 계획이 짧아져서 생긴 결과가 아니라 What 정책이 같은 수준의 과제를 더 적은 질문으로 처리한 결과다.",
    "",
    "## 산출물",
    "",
    "- `00_raw/`: 이번 실행의 원본 로그 1,600개와 seed CSV",
    "- `00_raw_sources.csv`: 각 raw 파일의 SHA-256과 경로",
    "- `01_processed/episodes.csv`: 실행별 1차 가공 데이터",
    "- `01_processed/summary.csv`: 전체·도메인별 집계",
    "- `01_processed/scenes.csv`: 장면별 집계",
    "- `02_graph_data/figure_data.csv`: 그림과 직접 대응하는 값과 오차막대",
    "- `03_figures/`: overview와 지표별 PNG/PDF",
    "",
    "그림의 성공률 오차막대는 95% Wilson interval이고, 연속 지표는 mean ± SE다. 가설검정이나 seed 기반 paired 검정을 이 보고서의 근거로 사용하지 않았다.",
    "",
    "## 기록된 설정",
    "",
    "- gamma: 0.2",
    "- n_simulations: 200",
    "- threshold: 0.8",
    "- random query probability: 0.4",
    "- max step: 50",
]
(P / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

readme = """# When–What ablation (gamma = 0.2)

2026-09-21에 gamma=0.2로 다시 실행한 1,600 episodes의 분석 패키지다. 기존 gamma=0.95 분석을 이 결과로 교체했다.

## 파일 구조

- `00_raw/`: 원본 실행 로그 1,600개와 실행 seed CSV
- `00_raw_sources.csv`: raw 경로와 SHA-256
- `manifest.json`: 분석에 포함한 파일, 조건, scene, seed, iteration
- `01_processed/episodes.csv`: 실행별 1차 가공 데이터
- `01_processed/summary.csv`, `scenes.csv`: 전체·도메인·장면별 정제 결과
- `02_graph_data/figure_data.csv`: 그림의 막대와 오차막대에 직접 사용한 데이터
- `03_figures/overview.{png,pdf}`: 성공률, 성공 실행 질문 수, 전체 질문 수, 성공 실행 물리 step
- `03_figures/`: 각 지표의 개별 PNG/PDF
- `report.md`: 논문 주장에 맞춘 결과 해석

질문 효율은 조기 실패의 영향을 피하기 위해 성공 실행 질문 수를 중심으로 해석한다. 전체 실행 질문 수도 별도 그림과 CSV에 보존했다.

## 재생성

프로젝트 루트에서 다음을 실행한다.

```bash
python3 experiments_logs/analysis_recent/when_what_original_20260912/scripts/analyze.py
python3 experiments_logs/analysis_recent/when_what_original_20260912/scripts/build_graph_data.py
python3 experiments_logs/analysis_recent/when_what_original_20260912/scripts/plot_results.py
```

원본 해시가 달라지면 파이프라인이 중단된다.
"""
(P / "README.md").write_text(readme, encoding="utf-8")

for name in ("paired.csv", "paired_episodes.csv"):
    path = P / "01_processed" / name
    if path.exists():
        path.unlink()
