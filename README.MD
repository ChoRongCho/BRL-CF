
# Meta-Confidence Simulation for Predicate-Level Models

## Overview

This project implements a predicate-level meta-confidence update framework with:

* Knowledge Base (KB) storing fact-level confidence
* Action-based triggering
* Bayesian update at fact level
* Beta-distributed meta-confidence at predicate level
* Strong confidence propagation across same-predicate facts
* Convergence visualization against ground-truth model confidence

---

## Architecture

### 1. KnowledgeBase (KB)

* Stores facts as:

```python
{ fact_name: confidence ∈ [0,1] }
```

* Predicate = substring before first `_`
* Strong propagation assumption:

  * All facts under same predicate share identical confidence after update.

---

### 2. FeedbackManager

Each timestep:

1. Compute action confidence:
   [
   P = \prod_i b(p_i)
   ]

2. Trigger with probability:
   [
   1 - P
   ]

3. For each precondition fact:

   * Sample internal logit:
     [
     l \sim U(0.5, 1.0)
     ]

   * Oracle observation:
     [
     o \sim Bernoulli(GT_MODEL_CONFIDENCE[predicate])
     ]

   * Bayesian update:

     [
     c = P(p|o)
     ]

4. Predicate-level update (pessimistic):

   * For same predicate within timestep:

     * Use **minimum posterior**
   * Update Beta:

     [
     \alpha \leftarrow \alpha + \lambda c
     ]
     [
     \beta \leftarrow \beta + \lambda(1-c)
     ]

5. Strong propagation:

   * Sample external confidence:
     [
     \theta \sim Beta(\alpha, \beta)
     ]
   * Average over multiple samples (variance reduction)
   * Overwrite all KB facts of that predicate with θ

---

## Meta-Confidence Model

Each predicate maintains:

```
meta_ab[predicate] = [alpha, beta]
```

Initialization:

[
\alpha = c,\quad \beta = 1-c
]

Where `c` is initial KB confidence.

---

## Ground Truth Model Confidence

```
GT_MODEL_CONFIDENCE = {
    "at": 0.99,
    "pose": 0.99,
    "predidcate1": 0.3,
    "predidcate2": 0.4,
    "predidcate3": 0.2,
    "ripe": 0.8,
    "unripe": 0.85,
}
```

Oracle returns True with probability equal to GT value.

---

## Convergence Behavior

* Running >1000 timesteps shows convergence
* Predicate confidence approaches GT
* Variance decreases as α+β increases
* Multi-sample averaging stabilizes propagation

---

## Visualization

For each predicate:

* Plot KB confidence over time
* Overlay GT as horizontal dashed orange line

Saved as:

```
plots/{predicate}_confidence.png
```

---

## Key Design Decisions

* Predicate-level meta model (not fact-level)
* Strong propagation
* Pessimistic intra-timestep aggregation (min posterior)
* Sampling-based propagation (not mean)
* Multiple Beta samples averaged per timestep

---

## Status

Algorithm complete.
Empirical convergence observed.
Ready for experimental analysis or ablation study.

