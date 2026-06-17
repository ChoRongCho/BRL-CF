# BRL-CF

This repository contains a partially observable robot task-planning framework based on POMCP, belief updates, and confidence-triggered user feedback. The active experimental domains are:

- `tomato`: harvest ripe tomatoes and discard rotten tomatoes under hidden ripeness/location information.
- `wastesorting`: sort waste objects into correct bins under hidden category/occlusion information.

The repository also includes a KnowNo-style LLM baseline and evaluation scripts for system experiments and user-study results.

## Project Structure

```text
.
  main.py                         # Threshold-based POMCP + feedback runner
  when_main.py                    # Query-timing policy ablation runner
  run/                            # Shell entrypoints for experiments
  scripts/
    domain/                       # Domain YAMLs: scenes, domain rules, robot skills
      tomato/
      wastesorting/
    environments/                 # Environment execution and state synchronization
    models/                       # Belief, transition, observation, reward, action/state models
      tomato/
      wastesorting/
    planners/                     # POMCP and search-tree implementation
    utils/                        # Argument parsing, ASP bridge, logging helpers
    baseline/                     # KnowNo baseline code, prompts, LLM settings
  experiments/
    system_eval/                  # System experiment analysis pipeline
    user_eval/                    # User-study analysis pipeline
  experiments_logs/
    system_log/                   # Raw system experiment logs
    user_log/                     # Raw user-study data
```


## Core Method

The main planner treats task execution as online planning under partial observability.

At each step:

1. POMCP searches over the current belief and selects an executable robot action.
2. The environment samples the transition outcome and observation.
3. The belief is updated using transition and observation likelihoods.
4. A confidence score is computed from the normalized entropy of belief-frontier weights.
5. If feedback is triggered, the system asks about a high-value symbolic fact, filters the belief, and replans.
6. The most likely frontier state becomes the current symbolic knowledge state.

The feedback mechanism is designed to ask only when uncertainty is operationally useful. The active query-timing policies are:

| Policy | Meaning |
|---|---|
| `no` | Never ask. |
| `all` | Ask at every step. |
| `ours` | Ask only when confidence is below the threshold. |
| `random` | Ask with a fixed random probability. |
| `knowno_gpt4` | KnowNo baseline using GPT-4-family settings. |
| `knowno_gpt35turbo` | KnowNo baseline using GPT-3.5-turbo-instruct settings. |

## Domains

### Tomato

The tomato domain contains one robot, four tomatoes, one dock station, and two stems: `stem_01` and `stem_02`. Hidden information includes tomato ripeness and, depending on scene setup, object state/location knowledge.

Representative actions:

- `navigate`
- `detect`
- `pick`
- `scan`
- `place`
- `discard`

The robot should place ripe tomatoes and discard rotten tomatoes. Scan labels are restricted to `ripe` and `rotten` in the KnowNo baseline environment.

### Waste Sorting

The waste domain contains one robot, four waste objects, and bins for `plastic`, `can`, `paper`, and `general` waste. Hidden information includes waste category and scene-specific occlusion.

Representative actions:

- `detect_waste`
- `pick`
- `place_plastic_bin`
- `place_can_bin`
- `place_paper_bin`
- `place_gw_bin`

For `scene_02` and `scene_03`, `waste_04` is initially under `waste_03` and is not detectable until `waste_03` is moved. For `scene_04` and `scene_05`, `waste_03` and `waste_04` are initially under `waste_01` and `waste_02`.

## Scripts Structure

### `scripts/domain`

Each domain has:

- `scene_XX.yaml`: initial/true hidden state for a scene.
- `domain_rule.yaml`: symbolic domain rules.
- `robot_skill.yaml`: action schemas and executable skills.

### `scripts/environments`

Environment code applies the selected action, samples observations, checks terminal conditions, and keeps symbolic knowledge synchronized with sampled hidden state.

### `scripts/models`

This folder contains shared model classes and domain-specific implementations:

- transition models
- observation models
- reward models
- belief/frontier state classes
- action representations

### `scripts/planners`

Contains the POMCP implementation and search tree. The planner uses UCB-style action selection during simulations:

```text
Q(h, a) + c * sqrt(log(N(h)) / N(h, a))
```

### `scripts/baseline`

Contains the KnowNo-style LLM baseline:

- `knowno_baseline_experiment.py`: unified baseline entrypoint.
- `run_knowno_baseline_gui.py`: GUI wrapper for baseline runs.
- `scripts/knowno_multistep_tomato.py`: tomato baseline runner.
- `scripts/knowno_multistep_waste.py`: waste baseline runner.
- `scripts/structured_prompts.py`: v1 prompt definitions.
- `scripts/structured_prompts_v2.py`: v2 prompt definitions.
- `llm_setting.json`: model/API configuration.

KnowNo logs are separated by model family:

```text
experiments_logs/system_log/<domain>/scene_XX_step50/when_knowno_gpt4/
experiments_logs/system_log/<domain>/scene_XX_step50/when_knowno_gpt35turbo/
```

## Run Files

Run commands from the repository root.

### `run/when_experiments.sh`

Runs one or more episodes of the query-timing ablation with `when_main.py`.

Example:

```bash
./run/when_experiments.sh \
  --domain tomato \
  --scene 1 \
  --iter 10 \
  --strategy ours \
  --max-step 50 \
  --seed random
```

Output:

```text
experiments_logs/system_log/<domain>/scene_XX_step50/when_<strategy>_rand_<prob>/
```

For `random`, set the trigger probability:

```bash
./run/when_experiments.sh --domain wastesorting --scene 3 --strategy random --random-query-prob 0.5
```

### `run/iterate_when.sh`

Runs the query-timing ablation over both domains, five scenes, and all four policies:

```text
all, no, ours, random
```

It archives existing matching logs, runs repeated experiments, then executes:

```bash
python3 experiments/system_eval/analysis_experiment.py
python3 experiments/system_eval/read_csv_experiment.py
```

### `run/run_threshold_experiment.sh`

Runs threshold-based feedback experiments with `main.py`.

Example:

```bash
./run/run_threshold_experiment.sh \
  --domain tomato \
  --scene 2 \
  --iter 10 \
  --threshold 0.8
```

Output:

```text
experiments_logs/system_log/<domain>/scene_XX_step50/thres_<threshold>/
```

### `run/iterate_th.sh`

Runs threshold sweeps over configured thresholds and scenes, then runs the current system-analysis pipeline.

### `run/run_knowno_baseline.sh`

Runs one KnowNo baseline episode through `scripts/baseline/knowno_baseline_experiment.py`.

Default values can be overridden by environment variables:

```bash
DOMAIN=tomato SCENE=01 PROMPT_VERSION=v2 DRY_RUN=true ./run/run_knowno_baseline.sh
```

Default tomato settings include:

```text
qhat = 0.92
temperature = 3.0
max_steps = 50
```

The output directory is selected from the model in `scripts/baseline/llm_setting.json`.

### `run/run_knowno_baseline_gui.sh`

Launches the GUI wrapper for KnowNo baseline execution.

```bash
./run/run_knowno_baseline_gui.sh
```

## Experiment Logs

Raw logs are stored under:

```text
experiments_logs/system_log/
```

Typical layout:

```text
experiments_logs/system_log/
  tomato/
    scene_01_step50/
      when_all_rand_0-5/
      when_no_rand_0-5/
      when_ours_rand_0-5/
      when_random_rand_0-5/
      when_knowno_gpt4/
      when_knowno_gpt35turbo/
  wastesorting/
    scene_01_step50/
      ...
```

User-study raw data is stored under:

```text
experiments_logs/user_log/
```

The old root-level `logs` and user raw-data paths are kept only as compatibility links when present.

## System Evaluation Pipeline

The active system-evaluation pipeline is in:

```text
experiments/system_eval/
```

It follows a three-stage structure:

1. `analysis_experiment.py`: raw text logs to normalized run-level CSV.
2. `read_csv_experiment.py`: normalized CSV to final plotting CSV.
3. `plot_figure.py`: final plotting CSV to figures.

Run the full pipeline:

```bash
python3 experiments/system_eval/analysis_experiment.py
python3 experiments/system_eval/read_csv_experiment.py
python3 experiments/system_eval/plot_figure.py --csv all
```

Outputs:

```text
experiments/system_eval/data/raw_runs/
experiments/system_eval/data/policy_compare_total.csv
experiments/system_eval/figure/00_YYYYMMDD_HHMMSS/
```

The generated comparison CSV uses policy names as x-axis conditions and domain suffixes as columns:

```text
all_waste, all_tomato, all_all,
no_waste, no_tomato, no_all,
ours_waste, ours_tomato, ours_all,
random_waste, random_tomato, random_all,
knowno_gpt4_waste, knowno_gpt4_tomato, knowno_gpt4_all,
knowno_gpt35turbo_waste, knowno_gpt35turbo_tomato, knowno_gpt35turbo_all
```

Older scripts are kept under `experiments/system_eval/legacy/` for reference only.

## User Evaluation Pipeline

User-study analysis is organized under:

```text
experiments/user_eval/
```

Raw user data stays in:

```text
experiments_logs/user_log/
```

Generated CSVs and figures should stay inside:

```text
experiments/user_eval/data/
experiments/user_eval/figure/
```

## Requirements

Install Python dependencies from the repository root:

```bash
conda activate myenv
pip install -r requirements.txt
```

The `requirements.txt` file covers the core planner, KnowNo baseline, plotting/analysis scripts, and optional tabletop simulation dependencies.

The KnowNo GUI also requires `tkinter`. If `import tkinter` fails, install it through conda or the OS package manager:

```bash
conda install tk
# or, on Ubuntu/Debian:
sudo apt-get install python3-tk
```

KnowNo baseline execution requires a valid OpenAI-compatible API setting in:

```text
scripts/baseline/llm_setting.json
```

## Quick Commands

Run one threshold experiment:

```bash
./run/run_threshold_experiment.sh --domain tomato --scene 1 --iter 1 --threshold 0.8
```

Run one query-timing experiment:

```bash
./run/when_experiments.sh --domain wastesorting --scene 1 --iter 1 --strategy ours --seed random
```

Check KnowNo command without calling the API:

```bash
DRY_RUN=true DOMAIN=tomato SCENE=01 PROMPT_VERSION=v2 ./run/run_knowno_baseline.sh
```

Regenerate system CSVs and figures:

```bash
python3 experiments/system_eval/analysis_experiment.py
python3 experiments/system_eval/read_csv_experiment.py
python3 experiments/system_eval/plot_figure.py --csv all
```
