# BRL-CF POMCP Experiments

This repository implements a POMCP-based reactive planner for partially observable robot task-planning domains. The active experimental domains are:

- `tomato`: tomato harvesting with hidden tomato ripeness and location.
- `wastesorting`: waste sorting with hidden waste category.

`blocksworld` files are present, but the current transition and observation implementations are incomplete and should not be treated as a validated experiment target.

## Repository Layout

- `main.py`: default POMCP experiment loop with confidence-threshold feedback.
- `when_main.py`: question-timing ablation loop for `no`, `all`, `ours`, and `random` query policies.
- `utils/arguments.py`: shared command-line arguments.
- `environments/env.py`: environment, transition execution, observation sampling, goal checking, and model synchronization.
- `models/`: belief update, transition, observation, reward, action, and state implementations.
- `models/tomato/`: tomato-domain transition, observation, and reward functions.
- `models/wastesorting/`: waste-sorting transition, observation, and reward functions.
- `planners/pomcp.py`: POMCP search implementation.
- `planners/tree.py`: action-observation tree implementation.
- `domain/<domain>/`: domain rules, scenes, and robot skill YAML files.
- `run/run_threshold_experiment.sh`: threshold sweep helper for `main.py`.
- `run/when_experiments.sh`: query-timing strategy helper for `when_main.py`.
- `run/iterate_th.sh`: loops over multiple thresholds and scenes, then runs analysis.
- `logs/`: default experiment log directory.

## Requirements

The code expects Python 3 and the following Python packages:

```bash
pip install numpy pyyaml clingo
```

If `clingo` is not installed, execution will fail during import of `utils.asp`.

## Quick Start

Run one default experiment:

```bash
python3 main.py \
  --domain wastesorting \
  --initial_state domain/wastesorting/scene_01.yaml \
  --threshold 0.8 \
  --max_step 50 \
  --log_dir logs/wastesorting/test
```

Run the question-timing ablation:

```bash
python3 when_main.py \
  --domain wastesorting \
  --initial_state domain/wastesorting/scene_01.yaml \
  --f_strategy 3 \
  --max_step 50 \
  --log_dir logs/wastesorting/when_ours
```

In `when_main.py`, the confidence threshold is fixed internally to `0.8`.

## Run Scripts

Threshold experiment:

```bash
./run/run_threshold_experiment.sh --scene 1 --iter 3 --threshold 0.8
```

Question-timing experiment:

```bash
./run/when_experiments.sh --scene 1 --iter 3 --strategy ours
./run/when_experiments.sh --scene 1 --iter 3 --strategy random --random-query-prob 0.3
./run/when_experiments.sh --scene 1 --iter 3 --strategy all
./run/when_experiments.sh --scene 1 --iter 3 --strategy no
```

The strategy names map to `f_strategy` as follows:

| Strategy | `f_strategy` | Meaning |
|---|---:|---|
| `no` | `1` | Never ask feedback questions. |
| `all` | `2` | Ask at every step. If triggered, at least one question is asked even when confidence is already above `0.8`. |
| `ours` | `3` | Ask only when confidence is below `0.8`. |
| `random` | `4` | Trigger asking with probability `--random_query_prob` regardless of confidence; default is `0.3`. If triggered, at least one question is asked. |

## Command-Line Arguments

All arguments are defined in `utils/arguments.py`.

### Domain And Files

| Argument | Default | Description |
|---|---:|---|
| `--domain` | caller default, usually `tomato` | Domain name. Active domains are `tomato` and `wastesorting`. |
| `--domain_rule` | `domain/<domain>/domain_rule.yaml` | Path to ASP-style domain rule YAML. |
| `--initial_state` | `domain/<domain>/scene_02.yaml` | Path to scene or initial-state YAML. |
| `--robot_skill` | `domain/<domain>/robot_skill.yaml` | Path to action schema YAML. |

### POMCP Settings

| Argument | Default | Description |
|---|---:|---|
| `--gamma` | `0.95` | Discount factor. |
| `--c` | `1.0` | UCB exploration constant. |
| `--max_depth` | `20` | Maximum simulation depth per rollout. |
| `--n_simulations` | `100` | Number of Monte Carlo simulations per planning step. |
| `--epsilon` | `0.005` | Rollout stops when `gamma ** depth < epsilon` after depth 0. |

The POMCP action selection score is:

```text
Q(h,a) + c * sqrt(log(N(h)) / N(h,a))
```

Unvisited action nodes receive infinite UCB score.

### Experiment Settings

| Argument | Default | Description |
|---|---:|---|
| `--seed` | `42` | Random seed. `when_main.py` applies this to `random` and `numpy`; `main.py` currently logs it but does not set global seeds. |
| `--max_step` | `20` | Maximum execution steps per episode. The shell scripts use `50`. |
| `--max_particles` | `100` | Logged belief particle cap argument. Note: the current POMCP tree uses `max_node_particles` if present, otherwise `128`; `--max_particles` is not directly used by `planners/tree.py`. |
| `--threshold` | `1.0` | Confidence threshold for feedback in `main.py`. Ignored by `when_main.py`, which fixes threshold to `0.8`. |
| `--log_dir` | `logs` | Directory for text logs produced by `utils.logger.logger_exp`. |

### Numeric Fluent Settings

These arguments are parsed and logged for numeric fluent experiments:

| Argument | Default | Description |
|---|---:|---|
| `--fluent_sample_sigma` | `0.05` | Gaussian support width for observed fluent particles. |
| `--pick_fluent_sigma` | `0.05` | Execution tolerance for comparing commanded and particle fluent values. |
| `--pick_success_rate` | `0.88` | Nominal pick success probability at the maximum-likelihood fluent command. |
| `--pick_success_floor` | `0.05` | Minimum pick success probability for distant fluent particles. |

### Feedback And Ablation Settings

| Argument | Default | Description |
|---|---:|---|
| `--f_strategy` | `1` | Feedback timing strategy: `1=no`, `2=all`, `3=ours`, `4=random`. Mainly used by `when_main.py`. |
| `--q_strategy` | `1` | Query-selection strategy placeholder. Current code uses information-gain fact selection. |
| `--answer_type` | `auto` | `auto` answers from hidden `true_init`; `human` prompts for `t/f` input. |
| `--random_query_prob` | `0.3` | Trigger probability for `f_strategy=4` in `when_main.py`. |

## Feedback Logic

Belief update uses transition and observation likelihoods:

```text
b'(s') proportional to T(s, a, s') * O(o | s', a)
```

Confidence is entropy-normalized:

```text
confidence = 1 - H(weights) / log2(number_of_frontiers)
```

When feedback is triggered, the planner selects the fact with maximum expected entropy reduction, asks whether it is true, filters the frontier states by the answer, and normalizes the remaining weights. In `auto` mode, the answer is generated from the scene file's `true_init` facts.

## Logs

Logs are written as text files by `utils/logger.py`. Each log contains:

- metadata: domain, scene path, threshold, seed, POMCP parameters, and log directory;
- plan summary: success flag, number of steps, cumulative reward, and total questions;
- timing: total and average search, execute, update, interaction, pruning, and wall-clock time;
- full plan: action per step with timing and reward;
- questions: fact queried, answer, confidence before and after;
- final knowledge facts and fluents.

Example log locations:

```text
logs/wastesorting/scene_01_step50/thres_0-8/
logs/wastesorting/scene_01_step50/when_random_thres_0.8_rand_0-3/
```

## Domain Summary

### Tomato

The tomato domain contains one robot, four tomatoes, one dock station, and two stems. Hidden state includes each tomato's stem location and ripeness class. Actions are `navigate`, `detect`, `pick`, `scan`, `place`, and `discard`.

Sparse rewards:

- reward `5` for successfully loading an observed and scanned ripe tomato;
- reward `5` for successfully discarding an observed and scanned rotten tomato;
- reward `0` otherwise.

### Waste Sorting

The waste-sorting domain contains one robot, four waste objects, and four bins: plastic, can, paper, and general waste. Hidden state is each waste object's category. Actions are `detect_waste`, `pick`, `place_plastic_bin`, `place_can_bin`, `place_paper_bin`, and `place_gw_bin`.

Sparse rewards:

- reward `5` for placing a waste object into the bin matching its category;
- additional reward `10` when the full scene goal becomes satisfied for the first time;
- reward `0` otherwise.

## POMCP Loop

At each execution step:

1. POMCP selects an applicable action from the current symbolic knowledge state.
2. The environment samples the transition and observation.
3. The belief frontier is updated using the transition probability and observation likelihood.
4. Optional feedback refines the frontier.
5. The most likely frontier is promoted to current knowledge.
6. The search tree is pruned to the executed action-observation branch.

This loop lets the planner replan online under partial observability instead of committing to a fixed offline plan.
