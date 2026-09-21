# State-CP calibration data

Generate the held-out dataset and qhat files with:

```bash
./run/build_state_cp_calibration.sh
```

The default run creates 25 independent seeded episodes for each of five scenes
and each domain. Exactly one ambiguous post-observation belief context is
sampled per episode. The 125 records per domain are shuffled by episode into
100 calibration and 25 test records. Evaluation experiment seeds must not be
reused here.

Files produced by the command:

- `state_cp_dataset.json`: joint belief hypotheses, coherent symbolic truth assignment,
  split, seed, scene, action, and observation.
- `state_cp_records.csv`: flat audit table.
- `state_cp_qhat.json`: domain qhats and held-out coverage/set-size diagnostics.

The State-CP class is the complete joint truth assignment over every symbolic
fact that differs across the current belief frontier. This includes
`detected`, `observed`, and `scanned` whenever they differ between particles.
The nonconformity score is
`1 - p_b(true assignment)`. The runtime question content remains Ours EIG, so
the experiment changes When while holding What fixed. The finite-sample rank is
`ceil((n+1) * target_coverage)`, adapted from KnowNo's split-conformal
calibration. KnowNo's action-option qhats cannot be reused.

## Current 95% calibration

The generated dataset was collected with gamma `0.2`, 100 POMCP simulations,
and 25 episodes per scene. Its calibrated thresholds are:

| Domain | qhat | Held-out coverage | Truth in belief support | Mean set size | Query trigger rate |
|---|---:|---:|---:|---:|---:|
| tomato | 0.9525166623 | 1.00 | 1.00 | 1.96 | 0.40 |
| wastesorting | 0.9825743984 | 1.00 | 1.00 | 1.48 | 0.16 |

The existing action-context-aware domain oracle grounds the externally
meaningful task facts. Calibration then selects the unique complete belief
hypothesis consistent with those oracle answers, preserving the same full
state representation used at runtime. It never fabricates independent values
for bookkeeping facts. A zero-match context is recorded as out of support; a
multiple-match context is rejected instead of guessed. In the generated 250
records every oracle assignment resolved to one unique full hypothesis and had
positive belief mass. Waste produces multiple-hypothesis prediction sets in 4
of 25 held-out contexts, giving a query-trigger rate of 0.16.
