# Experiment runners

These scripts follow `10_TODO_20260727.md` directly:

| Script | Experiment | Status |
|---|---|---|
| `e1_threshold.sh` | E1 threshold sweep, Ours + Oracle | runnable |
| `e2_ablation.sh` | E2 Ours / Ours-Random-When / Ours-Random-What, all Oracle | runnable |
| `e3_baselines.sh` | E3 Ours / KnowNo / Active Search, all Oracle | runnable |
| `e4_feedback.sh` | E4 Ours/KnowNo × VLM/Human | Human runnable interactively; GPT VLM placeholder |

Common defaults are in `common.sh`. Override them with environment variables:

```bash
E1_ITERATIONS=1 E1_DOMAINS=tomato E1_SCENES="1 3" THRESHOLDS="0.7 0.8" ./run/e1_threshold.sh
E2_ITERATIONS=1 E2_DOMAINS=tomato E2_SCENES="1 3" E2_METHODS="ours ours-random-when" ./run/e2_ablation.sh
E3_ITERATIONS=1 E3_DOMAINS=tomato E3_SCENES=1 E3_METHODS="ours knowno" ./run/e3_baselines.sh
E4_ITERATIONS=1 E4_DOMAINS=tomato E4_SCENES=1 E4_METHODS=ours E4_PROVIDERS=human ./run/e4_feedback.sh
```

Each E1-E4 runner updates one progress-bar line in an interactive terminal.
Redirected/non-interactive output prints only when the integer percentage
changes, so large runs do not produce one log line per episode:

```text
[###############---------------]  50% (30/60) E2 ours-random-when tomato scene=03
```

For non-interactive runs, Python stdout/stderr is written beside the result as
`console_seed_*.log` (or `run_*.console.log` for KnowNo), keeping the terminal
focused on progress. If a run fails, its last 40 console lines are printed.
Human-feedback runs remain interactive and therefore keep terminal output.

## Log layout

All new runs are stored below `experiments_logs/system_log` by default:

```text
e1_threshold/{domain}/scene_XX_step{max_step}/ours/oracle/tau_{value}/
e2_ablation/{domain}/scene_XX_step{max_step}/{method}/oracle/
e3_baselines/{domain}/scene_XX_step{max_step}/{method}/oracle/
e4_feedback/{domain}/scene_XX_step{max_step}/{method}/{vlm|human}/
```

Set `EXPERIMENT_LOG_ROOT` to place a run somewhere else without editing a
script. For example:

```bash
EXPERIMENT_LOG_ROOT=/tmp/brl_cf_smoke ITERATIONS=1 DOMAINS=tomato SCENES=1 \
  ./run/e1_threshold.sh
```

The older `00_ours`, `04_answer_mode`, `05_active_search`, and `06_knowno`
directories are legacy results. The new runners do not append to them.

Random-When uses the fixed validation query rates `0.48` for `tomato` and
`0.38` for `wastesorting`. They can be overridden with
`TOMATO_RANDOM_QUERY_PROB` and `WASTESORTING_RANDOM_QUERY_PROB`, respectively.

`BASE_SEED` defaults to `20260727`. All methods use the same deterministic seed
for a given domain, scene, and run index so their results can be paired. E4 is
interactive when `PROVIDERS=human`; keep `ITERATIONS` small while checking the
input flow. The default E4 provider list includes `vlm`, but that condition is
explicitly skipped until the shared GPT VLM provider is implemented.

E3 KnowNo uses GPT-4o calibration at score temperature `5.0` and target success
`0.85`: qhat `0.7779` for tomato and `0.7369` for wastesorting. Override with
`KNOWNO_SETTINGS`, `KNOWNO_SCORE_TEMPERATURE`, `KNOWNO_QHAT_TOMATO`, and
`KNOWNO_QHAT_WASTESORTING`.

`active-search` is the grounded-fact Adapted Attr-POMDP query-action planner
under `scripts/baseline/attr_pomdp/`. Result paths use the canonical name
`active-search`; metadata also records `implementation=adapted_attr_pomdp`, the
reference paper, and the grounded Boolean fact adaptation.

Noisy-Oracle and scale experiments were removed from `run/`: the former is
appendix-only in the experiment plan, and the latter is not an E1-E4 axis.
