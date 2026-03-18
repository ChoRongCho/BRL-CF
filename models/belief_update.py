from __future__ import annotations
import numpy as np


"""
Particle Belief Approximation for POMDP
=======================================

Belief update (exact):
    b_t(s_t) = eta * O(o_t | s_t, a_{t-1}) * sum_{s_{t-1}} T(s_t | s_{t-1}, a_{t-1}) b_{t-1}(s_{t-1})

where
    - b_t(s_t): posterior belief at time t
    - T(s_t | s_{t-1}, a_{t-1}): transition model
    - O(o_t | s_t, a_{t-1}): observation likelihood
    - eta: normalization constant

Particle approximation:
    b_t(s) ≈ sum_{i=1}^N w_t^(i) * delta(s - s_t^(i))

Bootstrap particle filter:
    1) Sample:
         s_t^(i) ~ T(. | s_{t-1}^(i), a_{t-1})
    2) Weight:
         w_t^(i) ∝ O(o_t | s_t^(i), a_{t-1})
    3) Normalize:
         w_t^(i) <- w_t^(i) / sum_j w_t^(j)
    4) Resample if needed:
         draw N particles from categorical(w_1, ..., w_N)

More general importance sampling form:
    if s_t^(i) ~ q(s_t | s_{t-1}^(i), a_{t-1}, o_t),
    then
         w_t^(i) ∝ w_{t-1}^(i) *
                    [ O(o_t | s_t^(i), a_{t-1}) * T(s_t^(i) | s_{t-1}^(i), a_{t-1}) ]
                    / q(s_t^(i) | s_{t-1}^(i), a_{t-1}, o_t)

In this rough implementation below, we use the bootstrap filter:
    q = T
so the weight simplifies to:
    w_t^(i) ∝ O(o_t | s_t^(i), a_{t-1})

ESS (effective sample size):
    ESS = 1 / sum_i (w_t^(i))^2

Resample when ESS < threshold.
"""

from models.state import State

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence
import copy
import math
import random


@dataclass
class Particle:
    state: State
    weight: float = 1.0


@dataclass
class Factored:
    state: State



@dataclass
class ParticleBelief:
    particles: List[Particle] = field(default_factory=list)

    def normalized_weights(self) -> List[float]:
        total = sum(p.weight for p in self.particles)
        if total <= 0:
            n = len(self.particles)
            return [1.0 / n] * n if n > 0 else []
        return [p.weight / total for p in self.particles]

    def effective_sample_size(self) -> float:
        ws = self.normalized_weights()
        if not ws:
            return 0.0
        return 1.0 / sum(w * w for w in ws)

    def estimate(self) -> Dict[str, Any]:
        """
        아주 러프한 belief summary.
        categorical / boolean / numeric 섞인 상태를 대충 aggregate.
        """
        if not self.particles:
            return {}

        ws = self.normalized_weights()
        keys = set()
        for p in self.particles:
            keys.update(p.state.keys())

        summary: Dict[str, Any] = {}

        for key in keys:
            values = [p.state.get(key, None) for p in self.particles]

            # bool
            if all(isinstance(v, bool) or v is None for v in values):
                prob_true = 0.0
                for p, w in zip(self.particles, ws):
                    if p.state.get(key, False) is True:
                        prob_true += w
                summary[key] = {
                    "type": "bernoulli",
                    "p_true": prob_true
                }
                continue

            # numeric
            if all(isinstance(v, (int, float)) or v is None for v in values):
                mean_val = 0.0
                total_w = 0.0
                for p, w in zip(self.particles, ws):
                    v = p.state.get(key, None)
                    if v is None:
                        continue
                    mean_val += w * float(v)
                    total_w += w
                if total_w > 0:
                    mean_val /= total_w
                summary[key] = {
                    "type": "numeric",
                    "mean": mean_val
                }
                continue

            # categorical / symbol-like
            counter: Dict[str, float] = {}
            for p, w in zip(self.particles, ws):
                v = p.state.get(key, None)
                name = str(v)
                counter[name] = counter.get(name, 0.0) + w
            summary[key] = {
                "type": "categorical",
                "distribution": dict(sorted(counter.items(), key=lambda x: -x[1]))
            }

        return summary


class BeliefUpdater:
    """
    Rough particle belief updater.

    state example:
        {
            "robot_at": "kitchen",
            "holding": None,
            "obj1_material": "plastic",
            "obj2_visible": True,
            "battery": 0.82
        }

    지금은 가능한 한 러프하게:
    - particle set 유지
    - action + observation으로 update
    - likelihood 기반 weight update
    - ESS 낮으면 resampling
    """

    def __init__(
        self,
        num_particles: int = 100,
        resample_threshold_ratio: float = 0.5,
        random_seed: Optional[int] = None,
    ) -> None:
        self.num_particles = num_particles
        self.resample_threshold_ratio = resample_threshold_ratio
        self.random = random.Random(random_seed)

        self.belief = ParticleBelief()
        self.time_step = 0

    def initialize(self, initial_particles: Optional[List[Dict[str, Any]]] = None) -> None:
        """
        사용자가 직접 초기 particle states를 넣어주는 방식.
        """
        self.time_step = 0
        self.belief.particles = []

        if not initial_particles:
            return

        n = len(initial_particles)
        w = 1.0 / n if n > 0 else 1.0

        for s in initial_particles:
            self.belief.particles.append(
                Particle(state=copy.deepcopy(s), weight=w)
            )

    def initialize_from_prior(self) -> None:
        """
        도메인 prior에서 particle을 샘플링.
        아직 러프 코드이므로 sample_initial_state()만 채우면 동작.
        """
        self.time_step = 0
        self.belief.particles = []

        for _ in range(self.num_particles):
            s0 = self.sample_initial_state()
            self.belief.particles.append(Particle(state=s0, weight=1.0 / self.num_particles))

    def update(self, action: Any, observation: Any) -> None:
        """
        Bootstrap particle filter:
            s_t^(i) ~ T(. | s_{t-1}^(i), a_{t-1})
            w_t^(i) ∝ O(o_t | s_t^(i), a_{t-1})
        """
        if not self.belief.particles:
            raise ValueError("Belief is empty. Initialize particles first.")

        self.time_step += 1

        new_particles: List[Particle] = []

        for old_particle in self.belief.particles:
            prev_state = old_particle.state

            predicted_state = self.transition_model(prev_state, action)
            likelihood = self.observation_likelihood(observation, predicted_state, action)

            # 수치 안정성 때문에 floor를 살짝 둠
            likelihood = max(likelihood, 1e-12)

            new_particles.append(
                Particle(
                    state=predicted_state,
                    weight=likelihood
                )
            )

        self.belief.particles = new_particles
        self.normalize_weights()

        ess = self.belief.effective_sample_size()
        threshold = self.resample_threshold_ratio * len(self.belief.particles)

        if ess < threshold:
            self.resample()

    def update_with_importance_sampling(self, action: Any, observation: Any) -> None:
        """
        좀 더 일반형.
        proposal q를 따로 정의하고 importance weight를 계산한다.

        w_t^(i) ∝ w_{t-1}^(i) *
                  [ O(o_t | s_t^(i), a_{t-1}) * T(s_t^(i) | s_{t-1}^(i), a_{t-1}) ]
                  / q(s_t^(i) | s_{t-1}^(i), a_{t-1}, o_t)

        지금은 러프하게 인터페이스만 잡아둠.
        """
        if not self.belief.particles:
            raise ValueError("Belief is empty. Initialize particles first.")

        self.time_step += 1
        new_particles: List[Particle] = []

        for old_particle in self.belief.particles:
            prev_state = old_particle.state
            prev_weight = old_particle.weight

            proposed_state = self.proposal_sample(prev_state, action, observation)

            obs_prob = max(self.observation_likelihood(observation, proposed_state, action), 1e-12)
            trans_prob = max(self.transition_probability(proposed_state, prev_state, action), 1e-12)
            proposal_prob = max(self.proposal_probability(proposed_state, prev_state, action, observation), 1e-12)

            new_weight = prev_weight * (obs_prob * trans_prob) / proposal_prob

            new_particles.append(
                Particle(
                    state=proposed_state,
                    weight=new_weight
                )
            )

        self.belief.particles = new_particles
        self.normalize_weights()

        ess = self.belief.effective_sample_size()
        threshold = self.resample_threshold_ratio * len(self.belief.particles)

        if ess < threshold:
            self.resample()

    def normalize_weights(self) -> None:
        total = sum(p.weight for p in self.belief.particles)
        n = len(self.belief.particles)

        if n == 0:
            return

        if total <= 0:
            uniform = 1.0 / n
            for p in self.belief.particles:
                p.weight = uniform
            return

        for p in self.belief.particles:
            p.weight /= total

    def resample(self) -> None:
        """
        multinomial resampling
        """
        if not self.belief.particles:
            return

        weights = [p.weight for p in self.belief.particles]
        states = [p.state for p in self.belief.particles]

        sampled_indices = self.random.choices(
            population=list(range(len(states))),
            weights=weights,
            k=len(states)
        )

        new_particles: List[Particle] = []
        uniform_w = 1.0 / len(states)

        for idx in sampled_indices:
            new_particles.append(
                Particle(
                    state=copy.deepcopy(states[idx]),
                    weight=uniform_w
                )
            )

        self.belief.particles = new_particles

    def estimate(self) -> Dict[str, Any]:
        return self.belief.estimate()

    def most_likely_particle(self) -> Optional[Particle]:
        if not self.belief.particles:
            return None
        return max(self.belief.particles, key=lambda p: p.weight)

    def sample_state(self) -> Optional[Dict[str, Any]]:
        if not self.belief.particles:
            return None
        weights = [p.weight for p in self.belief.particles]
        idx = self.random.choices(range(len(self.belief.particles)), weights=weights, k=1)[0]
        return copy.deepcopy(self.belief.particles[idx].state)

    def debug_print(self, top_k: int = 5) -> None:
        print(f"[BeliefUpdater] t={self.time_step}")
        print(f"num_particles={len(self.belief.particles)}")
        print(f"ESS={self.belief.effective_sample_size():.4f}")

        ranked = sorted(self.belief.particles, key=lambda p: p.weight, reverse=True)
        for i, p in enumerate(ranked[:top_k], start=1):
            print(f"\nParticle {i}")
            print(f"weight={p.weight:.6f}")
            print(p.state)

