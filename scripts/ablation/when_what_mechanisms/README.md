# Controlled When–What mechanism draft

This folder is an isolated implementation draft for four controlled conditions:

| Condition | When | What |
|---|---|---|
| `ours` | belief confidence threshold | expected information gain |
| `cp_when` | KnowNo-style CP action-set ambiguity | expected information gain |
| `value_when` | query-vs-physical root Q-value | expected information gain |
| `value_what` | belief confidence threshold | query root Q-value |

The implementation imports existing project modules read-only. It does not require changes to `main.py`, `scripts/models`, `scripts/baseline`, or existing `run/` scripts.

## Files

- `runner.py`: Ours physical-action loop with replaceable When/What policies.
- `policies.py`: four registered policy combinations.
- `query_episode.py`: shared Boolean question loop and stopping rule.
- `cp_when.py`: observation/belief-state adapter, LLM action scoring, and CP trigger.
- `value_evaluator.py`: restricted-root wrapper around the existing QaA POMCP planner.
- Each completed run writes `mechanism_trace.json` with every When decision,
  CP prediction set, EIG score, and root Q-value comparison.
- `run/run_when_what_mechanism.sh`: one episode.
- `run/iterate_when_what_mechanisms.sh` and `batch.py`: paired batch execution.
- `test_draft.py`: offline policy checks.

## Dry run

```bash
bash run/run_when_what_mechanism.sh \
  --condition value_when --domain tomato --scene 1 --seed 42 --dry-run

WW_ITERATIONS=2 bash run/iterate_when_what_mechanisms.sh --dry-run
```

## Small run

```bash
bash run/run_when_what_mechanism.sh \
  --condition ours --domain tomato --scene 1 --seed 42

bash run/iterate_when_what_mechanisms.sh --iter 2
```

Resume an interrupted timestamp directory explicitly:

```bash
bash run/iterate_when_what_mechanisms.sh \
  --resume --run-root experiments_logs/when_what_mechanisms/<timestamp>
```

`cp_when` calls the configured LLM twice at every physical step: once to generate action options and once to score them. The other three conditions do not make an LLM API call.

## Draft limitations to inspect before a full experiment

1. Confirm that the textual state adapter in `cp_when.py` expresses every domain fact needed by the current KnowNo prompts.
2. Inspect root Q-value visit counts in both domains. The evaluator reserves one initialization rollout and then enough simulations to visit every root candidate; a missing visit fails explicitly.
3. Run `--iter 2` and inspect individual logs before starting 40 repetitions.
