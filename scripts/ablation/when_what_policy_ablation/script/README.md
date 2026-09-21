# Controlled When–What policy ablation

The three entry points in the parent directory hold the experiment-facing
code. Shared implementation lives here.

| Entrypoint | When | What |
|---|---|---|
| `01_cp_when.py` | complete belief-state CP set has at least two states | Ours EIG inside that CP set |
| `02_value_when.py` | best query-action Q exceeds every physical-action Q | Ours EIG, exactly one question per physical step |
| `03_value_what.py` | Ours entropy-confidence threshold | highest query-action Q among ambiguous state facts |

Every query is one Boolean state-fact question. CP-When and Value-What
re-evaluate their When policy after each answer. Value-When uses the value
comparison only once per physical step and asks at most one Ours-EIG question.

State CP projects each belief particle onto every symbolic fact that differs
across the current frontier and treats the resulting complete joint truth
assignment as one hypothesis. This includes observation-bookkeeping predicates
such as `detected`, `observed`, and `scanned` when they are ambiguous. Its
nonconformity is `1 - p_b(true assignment)`. When the CP set has at least two
states, Ours EIG is computed after restricting and renormalizing the belief to
those CP-set states. The collector
samples one context per independent held-out episode. The calibrator uses
KnowNo's finite-sample split-conformal rank rule. The old KnowNo action-option
qhats are not valid here.

```bash
./run/build_state_cp_calibration.sh

STATE_CALIBRATION=scripts/ablation/when_what_policy_ablation/dataset/state_cp_qhat.json \
  bash run/run_when_what_policy_ablation.sh \
  --condition cp_when --domain tomato --scene 1 --seed 42

bash run/run_when_what_policy_ablation.sh \
  --condition value_when --domain tomato --scene 1 --seed 42

WW_STATE_CALIBRATION=scripts/ablation/when_what_policy_ablation/dataset/state_cp_qhat.json \
  bash run/iterate_when_what_policy_ablation.sh --iter 1
```

The original Random/Ours 2x2 experiment is in
`scripts/ablation/when_what_random/run_experiment.py` and continues to run via
`run/run_when_what_ablation.sh` and `run/iterate_when_what.sh`.
