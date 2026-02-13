
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import random
import math


# =========================
# 1) KB
# =========================

@dataclass
class KnowledgeBase:
    """Stores facts and their confidence values in [0, 1]."""
    facts: Dict[str, float] = field(default_factory=dict)

    def add_fact(self, fact: str, confidence: float) -> None:
        self.facts[fact] = self._clip01(confidence)

    def has_fact(self, fact: str) -> bool:
        return fact in self.facts

    def get_confidence(self, fact: str, default: float = 0.0) -> float:
        return float(self.facts.get(fact, default))
        
        
    def get_predicate_confidence(self, pred):
        vals = [v for k, v in self.facts.items() if k.split("_", 1)[0] == pred]
        if not vals:
            return float("nan")
        return sum(vals) / len(vals)
    
    
    def get_all_confidence(self) -> Dict[str, float]:
        """
        KB에 존재하는 모든 predicate에 대해
        {predicate: confidence} 형태로 반환.
        (같은 predicate의 fact들은 동일 값이라는 가정)
        """
        result: Dict[str, float] = {}

        for fact, conf in self.facts.items():
            predicate = fact.split("_", 1)[0]
            # 같은 predicate는 동일값이므로 그냥 덮어써도 무방
            result[predicate] = conf

        return result


    def set_confidence(self, fact: str, confidence: float) -> None:
        if fact not in self.facts:
            # If you want strict behavior, raise instead.
            self.facts[fact] = 0.0
        self.facts[fact] = self._clip01(confidence)

    def ensure_fact(self, fact: str, confidence: float = 0.0) -> None:
        """Create the fact if absent (useful for demos)."""
        if fact not in self.facts:
            self.add_fact(fact, confidence)

    def dump(self, prefix: str = "") -> List[Tuple[str, float]]:
        """Returns a sorted list of (fact, conf)."""
        items = sorted(self.facts.items(), key=lambda x: x[0])
        if prefix:
            items = [it for it in items if it[0].startswith(prefix)]
        return items
    
    
    def pretty_print(self) -> None:
        """
        Predicate별로 묶어서 보기 좋게 출력
        """
        if not self.facts:
            print("[KB] (empty)")
            return

        print("\n========== Knowledge Base ==========")

        # predicate 기준으로 그룹핑
        grouped: Dict[str, List[Tuple[str, float]]] = {}
        for fact, conf in self.facts.items():
            predicate = fact.split("_", 1)[0]
            grouped.setdefault(predicate, []).append((fact, conf))

        # predicate 정렬
        for pred in sorted(grouped.keys()):
            print(f"\n[{pred}]")
            for fact, conf in sorted(grouped[pred], key=lambda x: x[0]):
                print(f"  {fact:<35} : {conf:>6.3f}")

        print("====================================\n")

    @staticmethod
    def _clip01(x: float) -> float:
        if math.isnan(x):
            return 0.0
        return max(0.0, min(1.0, float(x)))
    
    
def generate_kb():
    KB = KnowledgeBase()
    KB.add_fact("ripe_tomato1", 0.5) # initial confidence setting
    KB.add_fact("ripe_tomato2", 0.5)
    KB.add_fact("unripe_tomato3", 0.5)
    KB.add_fact("at_stem1_tomato1", 0.5)
    KB.add_fact("at_stem1_tomato2", 0.5)
    KB.add_fact("at_stem1_tomato3", 0.5)
    KB.add_fact("at_robot1_stem1", 0.5)
    KB.add_fact("pose_tomato1_0.1_8.2_-0.1", 0.5)
    KB.add_fact("pose_tomato2_0.3_8.1_-0.2", 0.5)
    KB.add_fact("pose_tomato3_0.4_6.2_0.3", 0.5)
    KB.add_fact("predidcate1_dummy1", 0.5)
    KB.add_fact("predidcate1_dummy2", 0.5)
    KB.add_fact("predidcate2_dummy1", 0.5)
    KB.add_fact("predidcate2_dummy3", 0.5)
    KB.add_fact("predidcate3_dummy3", 0.5)

    return KB