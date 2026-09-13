# Query-as-Action (QaA) baseline

This baseline enumerates a fixed vocabulary of grounded Boolean queries as
actions and places them in the same action set as physical robot actions. We
refer to it as **Query-as-Action (QaA)**. POMCP is the solver used in the
current experiments.

The public action vocabulary is domain-specific. There is no generic
`Ask(fact)` action. Examples are:

```text
query_ripeness(tomato1)       -> ripe(tomato1)?
query_fresh(tomato1)          -> fresh(tomato1)?
query_robot_loc(brl_robot,stem_01) -> located(brl_robot,stem_01)?
query_holding(brl_robot,tomato1)   -> holding(brl_robot,tomato1)?
query_discarded(tomato1)      -> discarded(tomato1)?
```

Each observation is Boolean. A query has no physical transition, returns
`True` or `False`, and receives reward `-QUERY_COST`. After an executed query,
the existing Oracle supplies the answer, the particle belief is conditioned on
that answer, and its posterior MAP state becomes the symbolic knowledge base.
Query decisions do not increment the environment's physical-step counter.

Run one episode:

```bash
./scripts/baseline/targeted_query_pomdp/run.sh
DOMAIN=wastesorting SCENE=03 SEED=42 ./scripts/baseline/targeted_query_pomdp/run.sh
```

Run the default 2 domains x 5 scenes x 40 repetitions batch after editing the
configuration block at the top of `iterate.sh`:

```bash
./scripts/baseline/targeted_query_pomdp/iterate.sh
```

Important globals:

```text
DOMAIN, SCENE, SEED, MAX_STEP
N_SIMULATIONS, MAX_DEPTH
QUERY_COST, ANSWER_ACCURACY, MAX_CONSECUTIVE_QUERIES
LOG_ROOT
```

Logs are written below:

```text
experiments_logs/system_log/<domain>/scene_<NN>_step<max>/query_as_action/
```

`ANSWER_ACCURACY=1.0` uses the existing Boolean Oracle answer unchanged. A
smaller value applies the same answer-flip probability to both the POMCP query
observation model and the executed Oracle response.

The batch runner uses the paired seeds from the completed When--What
experiment, with `N_SIMULATIONS=100` and `QUERY_COST=1.0`.
