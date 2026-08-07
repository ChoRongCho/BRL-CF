from __future__ import annotations

from models.action import Action, normalize_fact
from models.state import State


class SymbolicWorld:
    """
    Explicit symbolic world state for real environment execution.

    The world owns the hidden true state.  It is updated only after an action is
    executed in the environment, using the actually sampled next state to tell
    whether each grounded add/delete effect happened.
    """

    def __init__(self, observable_init_state: State, true_init_state: State):
        self.observable_init_state = observable_init_state.copy()
        self.true_init_state = true_init_state.copy()
        self.initial_true_state = self._merge_initial_world(
            self.observable_init_state,
            self.true_init_state,
        )
        self.true_state = self.initial_true_state.copy()

    @staticmethod
    def _merge_initial_world(observable_state: State, true_state: State) -> State:
        merged = observable_state.copy()
        merged.merge_state(true_state)
        return merged

    def reset(self) -> State:
        self.true_state = self.initial_true_state.copy()
        return self.true_state

    def update_after_execution(self, action: Action | None, executed_state: State) -> State:
        if action is None:
            return self.true_state

        self._sync_executed_effects(action, executed_state)
        self._sync_executed_fluents(executed_state)
        self.after_action(action, executed_state)
        return self.true_state

    def _sync_executed_effects(self, action: Action, executed_state: State) -> None:
        for raw_fact in action.del_effects:
            fact = normalize_fact(raw_fact)
            if not executed_state.has_fact(fact):
                self.true_state.remove_fact(fact)

        for raw_fact in action.add_effects:
            fact = normalize_fact(raw_fact)
            if executed_state.has_fact(fact):
                self.true_state.add_fact(fact)

    def _sync_executed_fluents(self, executed_state: State) -> None:
        for obj, values in executed_state.fluents.items():
            for key, value in values.items():
                if float(value) != -1.0:
                    self.true_state.set_fluent(obj, key, value)

    def after_action(self, action: Action, executed_state: State) -> None:
        """Domain-specific hook."""


class TomatoWorld(SymbolicWorld):
    QUALITY_PREDICATES = {"ripe", "unripe", "rotten"}

    @staticmethod
    def _parse_fact(fact: str):
        fact = normalize_fact(fact)
        if "(" not in fact or not fact.endswith(")"):
            return fact, ()

        predicate, _, args = fact[:-1].partition("(")
        return predicate, tuple(args.split(",")) if args else ()

    def _true_quality_by_tomato(self) -> dict[str, str]:
        qualities = {}
        for fact in self.true_init_state.facts:
            predicate, args = self._parse_fact(fact)
            if predicate in self.QUALITY_PREDICATES and args:
                qualities[args[0]] = normalize_fact(fact)
        return qualities

    def after_action(self, action: Action, executed_state: State) -> None:
        # Observations may add a believed quality label to the executed
        # symbolic state. The hidden true world must keep the real label.
        true_quality = self._true_quality_by_tomato()

        for tomato, quality_fact in true_quality.items():
            for predicate in self.QUALITY_PREDICATES:
                self.true_state.remove_fact(f"{predicate}({tomato})")
            self.true_state.add_fact(quality_fact)


class WasteSortingWorld(SymbolicWorld):
    pass


class BlocksWorld(SymbolicWorld):
    pass


class KitchenWorld(SymbolicWorld):
    pass


class RoverWorld(SymbolicWorld):
    pass


class WateringWorld(SymbolicWorld):
    pass


def create_symbolic_world(
    domain: str,
    observable_init_state: State,
    true_init_state: State,
) -> SymbolicWorld:
    world_cls_by_domain = {
        "tomato": TomatoWorld,
        "wastesorting": WasteSortingWorld,
        "blocksworld": BlocksWorld,
        "kitchen": KitchenWorld,
        "rover": RoverWorld,
        "watering": WateringWorld,
    }

    try:
        world_cls = world_cls_by_domain[domain]
    except KeyError as exc:
        raise ValueError(f"Unknown symbolic world domain: {domain}") from exc

    return world_cls(
        observable_init_state=observable_init_state,
        true_init_state=true_init_state,
    )
