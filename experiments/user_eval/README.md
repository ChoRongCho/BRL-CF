# User Evaluation Pipeline

Run commands from the repository root:

```bash
cd /home/changmin/PyProject/00_BRL-CF
```

This folder follows the same high-level layout as `experiments/system_eval`:

```text
experiments/user_eval/
  *.py
  data/
  figure/
```

The raw user-study files remain outside this folder:

```text
experiments_logs/user_log/
```

The root `users_raw_data` and `user_raw_data` paths are kept as compatibility symlinks.

## Data Outputs

Generated CSV, TXT, MD, and ZIP outputs are stored under:

```text
experiments/user_eval/data/
```

Current data groups:

```text
data/01_op_sr/
data/02_time_consume/
data/03_user_survey/
data/domain_condition_friedman/
data/sagat_level_friedman/
```

## Figure Outputs

Generated figures are stored under:

```text
experiments/user_eval/figure/
```

Current figure groups:

```text
figure/01_op_sr/
figure/domain_condition_friedman/
figure/p_all_results/
figure/question_plots/
figure/sagat_level_friedman/
figure/timeline/
```

Additional scripts write new figures into the matching `figure/<analysis_name>/` folder by default.

## Main Scripts

Process survey responses:

```bash
python3 experiments/user_eval/process_responses.py
```

Run statistics:

```bash
python3 experiments/user_eval/run_statistics.py
```

Draw question plots:

```bash
python3 experiments/user_eval/visualize_question_plots.py
```

Generate scenario/condition accuracy CSVs:

```bash
python3 experiments/user_eval/scenario_condition_accuracy.py
```

Generate scenario/condition task-success CSVs:

```bash
python3 experiments/user_eval/scenario_condition_task_success.py
```

Draw aggregate P-all figures:

```bash
python3 experiments/user_eval/plot_p_all_results.py
```

Run domain-condition Friedman analysis:

```bash
python3 experiments/user_eval/analyze_domain_condition_friedman.py
```

Run SAGAT-level Friedman analysis:

```bash
python3 experiments/user_eval/analyze_sagat_level_friedman.py
```

## Path Rules

- Raw input stays in `experiments_logs/user_log/`.
- Generated CSV/TXT/MD/ZIP files go to `experiments/user_eval/data/`.
- Generated PNG/PDF files go to `experiments/user_eval/figure/`.
