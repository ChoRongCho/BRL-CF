# System Evaluation Pipeline

Run commands from the repository root:

```bash
cd /home/changmin/PyProject/00_BRL-CF
```

The active pipeline has three stages:

1. `analysis_*.py`: raw logs -> normalized run-level CSV.
2. `read_csv_*.py`: normalized CSV -> final plotting CSV.
3. `plot_figure.py --csv "file.csv"`: final plotting CSV -> figures.

## Layout

```text
experiments/system_eval/
  analysis_experiment.py
  read_csv_experiment.py
  plot_figure.py
  data/
  figure/
  legacy/
```

The active domain names are `tomato` and `wastesorting`. System logs are stored under `experiments_logs/system_log`; the root `logs` path is kept as a compatibility symlink.

## Stage 1

```bash
python3 experiments/system_eval/analysis_experiment.py
```

Inputs:

```text
experiments_logs/system_log/tomato/scene_XX_step50/*/*.txt
experiments_logs/system_log/wastesorting/scene_XX_step50/*/*.txt
experiments_logs/system_log/tomato/scene_XX_step50/when_knowno_gpt4/*.txt
experiments_logs/system_log/tomato/scene_XX_step50/when_knowno_gpt35turbo/*.txt
experiments_logs/system_log/wastesorting/scene_XX_step50/when_knowno_gpt4/*.txt
experiments_logs/system_log/wastesorting/scene_XX_step50/when_knowno_gpt35turbo/*.txt
```

Outputs:

```text
experiments/system_eval/data/raw_runs/tomato/raw_runs.csv
experiments/system_eval/data/raw_runs/wastesorting/raw_runs.csv
experiments/system_eval/data/raw_runs/domain_compare/raw_runs.csv
```

## Stage 2

```bash
python3 experiments/system_eval/read_csv_experiment.py
```

Default output:

```text
experiments/system_eval/data/policy_compare_total.csv
```

Default input:

```text
experiments/system_eval/data/raw_runs/domain_compare/raw_runs.csv
```

CSV shape:

```text
metric,metric_label,all_waste,all_tomato,all_all,no_waste,no_tomato,no_all,ours_waste,...
```

The condition prefix is the x-axis:

```text
all
no
ours
random
knowno_gpt4
knowno_gpt35turbo
```

Each condition has three comparison columns:

```text
waste
tomato
all
```

## Stage 3

Draw one CSV:

```bash
python3 experiments/system_eval/plot_figure.py \
  --csv experiments/system_eval/data/policy_compare_total.csv
```

Draw every CSV in `experiments/system_eval/data`:

```bash
python3 experiments/system_eval/plot_figure.py --csv all
```

Default output:

```text
experiments/system_eval/figure/00_YYYYMMDD_HHMMSS/
```

Each run creates a new timestamped folder and writes all generated png/pdf files there.

## Full Workflow

```bash
python3 experiments/system_eval/analysis_experiment.py
python3 experiments/system_eval/read_csv_experiment.py
python3 experiments/system_eval/plot_figure.py --csv all
```

## Legacy Scripts

Older one-off scripts are stored under:

```text
experiments/system_eval/legacy/
```

They are kept only for reference. New analysis should use the three-stage pipeline above.
