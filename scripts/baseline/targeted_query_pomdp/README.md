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

At every POMCP history node, symbolically applicable physical actions and
questions whose facts still have both true and false hypotheses share one
action set. If a physical action's hidden preconditions are false in a sampled
particle, that particle contributes a terminal reward of `-FAILURE_PENALTY` to
the action value. The default rollout policy samples only symbolically
applicable physical actions; query actions are evaluated in the explicit tree.

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
GAMMA, UCB_C, EPSILON
MAX_PARTICLES, MAX_BELIEF_PARTICLES, MAX_NODE_PARTICLES
QUERY_COST, FAILURE_PENALTY, ANSWER_ACCURACY
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
experiment, with `N_SIMULATIONS=200`, `QUERY_COST=1.0`, and
`FAILURE_PENALTY=10.0`. All POMCP search settings match Ours: depth `20`,
discount `0.95`, UCB exploration constant `1.0`, epsilon `0.005`, belief
particle cap `8000`, and node particle cap `8000`.
