# Experiment Analysis Code

This directory contains scripts for converting raw experiment logs into CSV
summaries and plotting those summaries.

## Workflow

Run from the repository root.

```bash
python3 exp_code/analysis_thres.py
python3 exp_code/analysis_when.py
python3 exp_code/read_csv_thres.py
python3 exp_code/read_csv_when_bar.py
```

The analysis scripts read raw logs from:

```text
logs/tomato
logs/wastesorting
```

The CSV summaries are written to:

```text
logs/<domain>/scene_metrics/
```

The figures are written to:

```text
figures/<domain>/plots_step50/
figures/<domain>/plots_step50_when/
```

## Scripts

- `analysis_common.py`
  - Shared parser and CSV aggregation helpers.
  - Extracts `[Plan Summary]`, `[Timing]`, and `[Action Schema Summary]`.

- `analysis_thres.py`
  - Reads raw threshold folders:
    `logs/<domain>/scene_XX_step50/thres_<threshold>`.
  - Aggregates thresholds that are present in all five scenes.
  - Writes CSV files to:
    `logs/<domain>/scene_metrics/step_50_thres_<threshold>/`.

- `analysis_when.py`
  - Reads raw question-policy folders:
    `logs/<domain>/scene_XX_step50/when_<strategy>_rand_0-3`.
  - Supports `all`, `no`, `ours`, and `random`.
  - Aggregates strategies that are present in all five scenes.
  - Writes CSV files to:
    `logs/<domain>/scene_metrics/step_50_thres_<strategy>/`.

- `read_csv_thres.py`
  - Reads numeric threshold CSV folders from `scene_metrics`.
  - Ignores strategy folders such as `step_50_thres_all`.
  - Generates per-scenario threshold plots for both domains.

- `read_csv_when_bar.py`
  - Reads strategy CSV folders from `scene_metrics`.
  - Generates strategy comparison bar plots for both domains.

## Expected Raw Log Format

Each raw log file should contain:

```text
[Plan Summary]
success: True
steps: 42
cumulated_reward: 12.5
total_questions: 3

[Timing]
total_time: 1.23s

[Action Schema Summary]
action_schema action_count question_count
...
```

Logs missing required summary fields are skipped in the aggregate values.

## Output CSV Files

Each output directory contains:

```text
success_rate.csv
average_step.csv
average_reward.csv
average_question.csv
elapsed_time.csv
action_schema_summary.csv
```

Metric CSV files contain:

- one column per scene folder,
- one row per raw log run,
- a `95% confidence interval` row,
- a final aggregate row.

`action_schema_summary.csv` contains action/query totals and expected questions
per executed action schema.
