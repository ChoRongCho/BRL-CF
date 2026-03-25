from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import re
import random

from models.state import State
from models.action import Action


@dataclass(slots=True)
class Observation:
    facts: State

    def is_empty(self) -> bool:
        return len(self.facts) == 0

    def __repr__(self) -> str:
        return f"Observation({self.facts})"


@dataclass(slots=True)
class ObservationOutcome:
    facts: List[str]
    probability: float



class ObservationModel:
    def __init__(
        self,
        domain: str,
        actions: List[Action],
        obj_type: Dict[str, List[str]],
        noise: float = 0.15
    ):
        self.domain = domain
        self.noise = noise
        self.obj_type = obj_type

        self.type_map = self._build_type_map()
        self.domain_model = self._build_domain_model()
        self.action_observation_space: Dict[str, List[str]] = {}

        self._build_from_actions(actions)

    def _build_type_map(self) -> Dict[str, List[str]]:
        type_map = {}
        for type_declare, objects in self.obj_type.items():
            m = re.match(r".*\(([A-Z])\)", type_declare)
            if m:
                type_symbol = m.group(1)
                type_map[type_symbol] = objects
        return type_map

    def _build_domain_model(self):
        if self.domain == "tomato":
            from models.tomato.obs import ObservationTomato
            return ObservationTomato(type_map=self.type_map, noise=self.noise)

        elif self.domain == "blocksworld":
            from models.blocksworld.obs import ObservationBlocksworld
            return ObservationBlocksworld(type_map=self.type_map, noise=self.noise)

        elif self.domain == "wastesorting":
            from models.wastesorting.obs import ObservationWastesorting
            return ObservationWastesorting(type_map=self.type_map, noise=self.noise)

        else:
            raise ValueError(f"Unknown domain: {self.domain}")

    def _build_from_actions(self, actions: List[Action]) -> None:
        for action in actions:
            self.action_observation_space[action.name] = self.domain_model.build_candidates(action)

    def get_observation_candidates(self, action: Action) -> List[str]:
        return self.action_observation_space.get(action.name, [])

    def get_observation_distribution(self, state: State, action: Action) -> List[ObservationOutcome]:
        return self.domain_model.get_observation_distribution(state, action)

    def sample(self, state: State, action: Action) -> Observation:
        outcomes = self.get_observation_distribution(state, action)

        if not outcomes:
            return Observation(State())

        r = random.random()
        cum = 0.0

        for outcome in outcomes:
            cum += outcome.probability
            if r <= cum:
                obs_state = State()
                for fact in outcome.facts:
                    obs_state.add_fact(fact)
                return Observation(obs_state)

        obs_state = State()
        for fact in outcomes[-1].facts:
            obs_state.add_fact(fact)
        return Observation(obs_state)


    def likelihood(self, observation: Observation, state: State, action: Action) -> float:
        outcomes = self.get_observation_distribution(state, action)
        
        obs_set = set(observation.facts.facts)
        
        # Debug        
        # print("[] Action: ", action)
        # print("[] outcomes: ", outcomes)
        # print("[] obs_set: ", obs_set)
        # print()

        for outcome in outcomes:
            if set(outcome.facts) == obs_set:
                return outcome.probability
        return self.noise
    
    
    
    def pretty_print_candidates(self, sort_keys: bool = True):
        print("\n========== Observation Candidates ==========\n")

        action_names = list(self.action_observation_space.keys())
        if sort_keys:
            action_names.sort()

        for action_name in action_names:
            candidates = self.action_observation_space[action_name]

            print(f"[ACTION] {action_name}")
            print(f"  #candidates: {len(candidates)}")

            for i, fact in enumerate(candidates):
                print(f"    ({i}) {fact}")

            print("-" * 50)

        print("\n============================================\n")
        
        
    def pretty_print_distribution(self, state: State, action: Action):
        print("\n========== Observation Distribution ==========\n")

        print(f"[ACTION] {action.name}")
        print(f"[STATE] {state}\n")

        outcomes = self.get_observation_distribution(state, action)

        if not outcomes:
            print("  (no outcomes)")
            return

        total_prob = sum(o.probability for o in outcomes)

        for i, o in enumerate(outcomes):
            print(f"  ({i}) p = {o.probability:.4f}")

            if o.facts:
                print("      facts:")
                for f in o.facts:
                    print(f"          {f}")
            else:
                print("      (empty)")

        print(f"\n  total_prob = {total_prob:.4f}")

        if abs(total_prob - 1.0) > 1e-6:
            print("  [WARNING] probabilities do not sum to 1.0")

        print("\n=============================================\n")