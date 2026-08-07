# System Evaluation Pipelines

The active E1-E4 pipelines are isolated by experiment:

```text
experiments/system_eval/
  e1/{analysis.py,read.py,plot.py}
  e2/{analysis.py,read.py,plot.py}
  e3/{analysis.py,read.py,plot.py}
  e4/{analysis.py,read.py,plot.py}
  run_e1.sh
  run_e2.sh
  run_e3.sh
  run_e4.sh
```

Run all three stages for one experiment:

```bash
./experiments/system_eval/run_e1.sh
./experiments/system_eval/run_e2.sh
./experiments/system_eval/run_e3.sh
./experiments/system_eval/run_e4.sh
```

Each runner executes `analysis.py`, `read.py`, and `plot.py` sequentially.
Outputs are grouped under the shared data and figure roots:

```text
data/eN/runs.csv
data/eN/summary.csv
figure/eN/YYYYMMDD_HHMMSS/*.png|*.pdf
```

Every metric is plotted three ways: `_all`, `_success_only`, and
`_failure_only`. The summary CSV records the same split in its `outcome` column.

The older unified pipeline documentation below is retained for compatibility.

Run commands from the repository root:

```bash
cd /home/changmin/PyProject/00_BRL-CF
```

The active pipeline has three stages:

1. `analysis_experiment.py`: raw logs -> normalized run-level CSV.
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

Default inputs:

```text
experiments_logs/system_log/04_answer_mode/{domain}/scene_XX_step50/answer_*.txt
experiments_logs/system_log/00_ours/{domain}/scene_XX_step50/ours_answer_*.txt
```

`04_answer_mode/**/answer_oracle_*` is excluded from aggregation. The oracle baseline is always read from `00_ours`.

Outputs:

```text
experiments/system_eval/data/raw_runs/00_ours/tomato.csv
experiments/system_eval/data/raw_runs/00_ours/wastesorting.csv
experiments/system_eval/data/raw_runs/00_ours/all_domains.csv
```

## Stage 2

```bash
python3 experiments/system_eval/read_csv_experiment.py
```

Default output:

```text
experiments/system_eval/data/refined/04_answer_mode.csv
```

Refined table outputs follow the numbered experiment index:

```text
experiments/system_eval/data/refined/01_threshold.csv
experiments/system_eval/data/refined/02_when.csv
experiments/system_eval/data/refined/03_scale.csv
experiments/system_eval/data/refined/04_answer_mode.csv
experiments/system_eval/data/refined/05_active_search.csv
experiments/system_eval/data/refined/06_knowno.csv
```

Default input:

```text
experiments/system_eval/data/raw_runs/04_answer_mode/all_domains.csv
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
  --csv experiments/system_eval/data/refined/04_answer_mode.csv
```

Draw every CSV in `experiments/system_eval/data/refined`:

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
python3 experiments/system_eval/plot_figure.py
```

Current defaults scan the numbered system-log experiment layout:

```text
experiments_logs/system_log/00_ours
experiments_logs/system_log/01_threshold
experiments_logs/system_log/02_when
experiments_logs/system_log/03_scale
experiments_logs/system_log/04_answer_mode
experiments_logs/system_log/05_active_search
experiments_logs/system_log/06_knowno
```

Missing numbered directories are skipped. `04_answer_mode/**/answer_oracle_*` is excluded; oracle reference rows come from `00_ours`.

The default outputs are:

```text
experiments/system_eval/data/raw_runs/04_answer_mode/all_domains.csv
experiments/system_eval/data/refined/04_answer_mode.csv
experiments/system_eval/figure/answer_mode/00_YYYYMMDD_HHMMSS/
```

## Legacy Scripts

Older one-off scripts are stored under:

```text
experiments/system_eval/legacy/
```

They are kept only for reference. New analysis should use the three-stage pipeline above.
