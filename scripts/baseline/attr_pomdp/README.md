# Attr-POMDP baseline

This directory contains an independent, paper-based reimplementation of
Attr-POMDP from:

> Yang Yang, Xibai Lou, and Changhyun Choi, "Interactive Robotic Grasping with
> Attribute-Guided Disambiguation," ICRA 2022, arXiv:2203.08037.

No official public source repository was found on the paper page, project page,
GitHub search, or CatalyzeX as of 2026-09-12. This code is therefore not the
authors' source code and must be described as a reimplementation.

## Paper model reproduced

- hidden discrete candidate state;
- deterministic hidden-state transition during dialogue;
- `AskAttr` cost `-0.1`;
- correct/incorrect terminal selection reward `+1/-1`;
- cooperative binary response model with probability `0.99`;
- Bayesian belief update;
- depth-3 belief-tree planning;
- optional `AskPoint` model with cost `-0.3` in `planner.py`.

## BRL adaptation

The paper selects one target object from a set of image detections. BRL instead
maintains candidate symbolic successor states after a robot action. The adapter
uses this direct correspondence:

| Attr-POMDP paper | BRL adapter |
| --- | --- |
| candidate target object | candidate frontier state |
| object attribute | discriminative symbolic fact |
| `AskAttr(attribute)` | ask whether a fact is true |
| `Grasp(candidate)` | commit the MAP frontier state |

The BRL experiment has no physical pointing question equivalent, so `AskPoint`
is excluded from the adapter action set. It remains implemented in the generic
planner for completeness. Answers come from the existing BRL Oracle without
modifying its domain rules.

A symbolic frontier may expose far more actions than the paper's small
color/location concept set. Before depth-3 search, the adapter therefore keeps
the eight facts with lowest expected posterior entropy. This computational cap
is configurable with `MAX_CANDIDATE_QUESTIONS` and is recorded in every log.

## Run

From the repository root:

```bash
scripts/baseline/attr_pomdp/run.sh
```

Configuration uses environment variables:

```bash
DOMAIN=wastesorting SCENE=03 SEED=1234 \
  scripts/baseline/attr_pomdp/run.sh
```

The main parameters default to the paper values and can be overridden with
`ATTR_DEPTH`, `ATTRIBUTE_COST`, and `ANSWER_ACCURACY`.

The full two-domain batch (five scenes and 40 runs per scene) is:

```bash
scripts/baseline/attr_pomdp/iterate.sh
```

Logs are written under:

```text
experiments_logs/system_log/<domain>/scene_<NN>_step50/attr_pomdp/
```

## Files

- `planner.py`: generic finite-horizon Attr-POMDP solver.
- `controller.py`: BRL frontier-state and Oracle adapter.
- `run_experiment.py`: isolated BRL experiment entry point.
- `run.sh`: convenience runner.
- `iterate.sh`: 400-run two-domain batch runner.
