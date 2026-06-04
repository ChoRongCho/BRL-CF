# User Study Response Processing

이 폴더는 사용자 응답 원본 CSV를 조건별 문항 통계 CSV와 그래프로 변환합니다.

## Files

- `answer_sheet.csv`: SAGAT 정답표입니다. `평가 도메인`, `시나리오`, `조건 선택`, `Q1`-`Q6` 컬럼을 사용합니다.
- `response.csv`: 사용자 응답 원본입니다.
- `process_responses.py`: `answer_sheet.csv`와 `response.csv`를 읽어 `output.csv` 하나를 생성합니다.
- `output.csv`: 조건별, 도메인별, 문항별 평균/표준편차/표본수 결과입니다.
- `metadata_summary.txt`: 전체 실험 N, 기간, 날짜별 피험자 수, 피험자별 조건 순서, 피험자별 SAGAT 정답률, `S x C x Domain` 출현 빈도 요약입니다.
- `visualize_question_plots.py`: `output.csv`만 읽어 `question_plots/`에 그래프를 생성합니다.
- `run_statistics.py`: `response.csv` 원자료를 사용해 반복측정 통계검정을 수행합니다.
- `stats_results.csv`: Friedman test와 planned Wilcoxon signed-rank test 결과입니다.
- `stats_summary.txt`: 논문 작성용 통계 결과 요약입니다.

## Usage

1. `answer_sheet.csv`와 `response.csv`를 같은 폴더에 둡니다.
2. 후처리 CSV를 생성합니다.

```bash
python3 process_responses.py
```

3. 그래프를 생성합니다.

```bash
python3 visualize_question_plots.py
```

4. 반복측정 통계검정을 수행합니다.

```bash
python3 run_statistics.py
```

## Notes

- 기본 입력은 `answer_sheet.csv`, `response.csv`이고 기본 출력은 `output.csv`, `metadata_summary.txt`입니다.
- SAGAT 그래프는 `answer_sheet.csv`의 `Q1`-`Q6` 정답값이 채워져 있을 때 생성됩니다.
- 그래프 생성 시 `question_plots/`의 기존 `*_condition_mean.png` 파일은 먼저 삭제됩니다.
- 통계검정은 피험자별 조건 평균을 만든 뒤 Friedman test와 C4 중심 planned exact Wilcoxon signed-rank test를 수행합니다.
