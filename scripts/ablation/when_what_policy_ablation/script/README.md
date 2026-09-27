# Controlled When–What policy ablation

The three entry points in the parent directory hold the experiment-facing
code. Shared implementation lives here.

| Entrypoint | When | What |
|---|---|---|
| `01_cp_when.py` | KnowNo action CP set is empty, has multiple actions, or contains the fallback option | Ours EIG over the current belief |
| `02_value_when.py` | best query-action Q exceeds every physical-action Q | Ours EIG, exactly one question per physical step |
| `03_value_what.py` | Ours entropy-confidence threshold | highest query-action Q among ambiguous state facts |

Every actual query is one Boolean state-fact question. In CP-When, KnowNo's
LLM action prediction set is used only as a query trigger; its action choice is
never executed. Once triggered, the same Ours EIG policy used by the reference
condition chooses the state fact to ask. The trigger is re-evaluated after each
answer.

The default action-CP calibration values match the KnowNo baselines:
`qhat=0.8404` for Tomato, `qhat=0.8704` for Waste Sorting, and score
temperature `5.0`.

```bash
bash run/run_when_what_policy_ablation.sh \
  --condition cp_when --domain tomato --scene 1 --seed 42

bash run/run_when_what_policy_ablation.sh \
  --condition value_when --domain tomato --scene 1 --seed 42

bash run/iterate_when_what_policy_ablation.sh --iter 1
```

The discarded belief-state CP experiment is preserved under
`belief_state_cp_archive/` and is not imported by these entry points.

The original Random/Ours 2x2 experiment is in
`scripts/ablation/when_what_random/run_experiment.py` and continues to run via
`run/run_when_what_ablation.sh` and `run/iterate_when_what.sh`.
