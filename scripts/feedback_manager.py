# scripts/feedback_manager.py

from scripts.knowledge_base import KnowledgeBase
from scripts.planner import Action

from typing import Dict, List, Tuple
import random


GT_MODEL_CONFIDENCE = {
    "at": 0.99,
    "pose": 0.99,
    
    # "predidcate1": 0.3,
    # "predidcate2": 0.4,
    # "predidcate3": 0.2,
    "predidcate1": 0.8,
    "predidcate2": 0.9,
    "predidcate3": 0.9,
    
    "ripe": 0.8,
    "unripe": 0.85,
}

PREDICATES = list(GT_MODEL_CONFIDENCE.keys())


class FeedbackManager:
    def __init__(self, 
                 lam: float = 1.0, 
                 n_sample=1,
                 seed: int = 43):
        
        # 글로벌 증거 강도(학습률)
        self.lam: float = lam

        # predicate -> [alpha, beta]
        # (리스트로 유지: a,b 갱신이 간단)
        self.meta_ab = None

        self.rng = random.Random(seed)
        self.missing_confidence: float = 0.0
        self.n_sample = n_sample
        
        self.number_of_query = 0

    # ---------- utilities ----------
    @staticmethod
    def predicate_of(fact: str) -> str:
        # 첫 '_' 전이 predicate
        return fact.split("_", 1)[0]

    def _ensure_predicate(self, pred: str) -> None:
        if pred not in self.meta_ab:
            self.meta_ab[pred] = [1.0, 1.0]

    def _beta_sample(self, a: float, b: float) -> float:
        return float(self.rng.betavariate(a, b))
    
    
    def init_beta_dist(self, kb: KnowledgeBase):
        init_confidence = kb.get_all_confidence()
        
        self.meta_ab = {}

        for pred, c in init_confidence.items():
            c = float(c)

            # 수치 안정성 (0,1 극단 방지)
            eps = 1e-6
            c = min(max(c, eps), 1.0 - eps)

            self.meta_ab[pred] = [c, 1.0 - c]

        print("[INIT BETA DISTRIBUTION]")
        for pred, (a, b) in self.meta_ab.items():
            print(f"  {pred:<12} -> alpha={a:.3f}, beta={b:.3f}")
        
        
        

    # ---------- core logic ----------
    def compute_action_confidence(self, action: Action, kb: KnowledgeBase) -> float:
        """action의 preconditions confidence 곱"""
        p = 1.0
        for fact in action.preconditions:
            if kb.has_fact(fact):
                c = kb.get_confidence(fact)
            else:
                c = self.missing_confidence

            p *= float(c)
            if p <= 0.0:
                return 0.0

        return max(0.0, min(1.0, p))

    def should_trigger_for_action(self, action: Action, kb: KnowledgeBase) -> bool:
        """trigger 확률 = 1 - Π(conf(preconditions))"""
        p = self.compute_action_confidence(action, kb)
        return self.rng.random() < (1.0 - p)

    def oracle_query(self, fact: str) -> bool:
        """
        외부 오라클(Query 모듈) 시뮬레이션.
        predicate별 GT 확률로 True(=맞다)를 반환.
        """
        pred = self.predicate_of(fact)
        gt = float(GT_MODEL_CONFIDENCE.get(pred, 0.5))
        return self.rng.random() < gt

    def bayesian_update(self, likelihood: float, prior: float, observation: bool) -> float:
        """
        posterior = P(p | o)
        likelihood = P(o=True | p=True) 같은 해석(실험용 internal logit)
        prior = P(p)
        """
        # 수치 안정성 (극단값 방지)
        eps = 1e-12
        prior = min(max(float(prior), eps), 1.0 - eps)
        likelihood = min(max(float(likelihood), eps), 1.0 - eps)

        if observation:  # o=True
            num = likelihood * prior
            den = likelihood * prior + (1.0 - likelihood) * (1.0 - prior)
        else:            # o=False
            num = (1.0 - likelihood) * prior
            den = (1.0 - likelihood) * prior + likelihood * (1.0 - prior)

        return num / den

    def update_beta_distribution(self, pred: str, c: float) -> Tuple[float, float]:
        """
        c = posterior confidence (0~1)
        alpha += lam*c
        beta  += lam*(1-c)
        """
        self._ensure_predicate(pred)
        a, b = self.meta_ab[pred]
        a = a + self.lam * float(c)
        b = b + self.lam * (1.0 - float(c))
        self.meta_ab[pred] = [a, b]
        return a, b

    def propagate_predicate_confidence(self, kb: KnowledgeBase, pred: str, n_samples: int = 10) -> float:
        """
        강한 전파 (덜 튐 버전):
        - pred의 Beta에서 theta를 n_samples번 샘플링
        - 그 평균(theta_mean)을 KB에 저장(해당 predicate의 모든 fact에 overwrite)
        """
        self._ensure_predicate(pred)
        a, b = self.meta_ab[pred]

        # 여러 번 샘플링해서 평균
        thetas = [self._beta_sample(a, b) for _ in range(max(1, int(n_samples)))]
        theta_mean = sum(thetas) / len(thetas)

        updated_facts = 0
        for fact in list(kb.facts.keys()):
            if self.predicate_of(fact) == pred:
                if hasattr(kb, "set_confidence"):
                    kb.set_confidence(fact, theta_mean)
                else:
                    kb.facts[fact] = theta_mean
                updated_facts += 1

        print(
            f"  pred={pred:<12} "
            f"theta_mean={theta_mean:>6.3f} "
            f"samples={len(thetas):02d} "
            f"updated_facts={updated_facts}"
        )

        return theta_mean

    def handle_action(self, action: Action, kb: KnowledgeBase) -> Dict:
        """
        action 1개 처리:
        - trigger 여부 판단
        - trigger면 모든 precondition fact에 대해 Query를 수행해 posterior를 계산
        - 같은 predicate는 timestep 내에 1번만(비관적 min posterior)로 Beta 업데이트
        - 업데이트된 predicate에 대해 강한 전파로 KB를 일괄 갱신
        """
        p = self.compute_action_confidence(action, kb)
        trigger = self.rng.random() < (1.0 - p)

        print(f"[FeedbackManager] action={action.name} preconf_product={p:.4f} trigger={trigger}")

        log = {
            "action_name": action.name,
            "preconditions": list(action.preconditions),
            "preconf_product": p,
            "trigger": trigger,
            "queries": [],
            "beta_updates": [],
            "kb_updates": [],
        }

        if not trigger:
            return log

        self.number_of_query += 1
        # predicate별로 "가장 낮은 posterior"만 유지 (비관적)
        min_post_by_pred: Dict[str, Tuple[float, str]] = {}

        # 1) fact 단위 posterior 계산
        for fact in action.preconditions:
            predicate = self.predicate_of(fact)
            internal_logit = self.rng.uniform(0.5, 1.0)

            if kb.has_fact(fact):
                prior = kb.get_confidence(fact)
            else:
                prior = self.missing_confidence

            obs = self.oracle_query(fact)

            posterior = self.bayesian_update(
                likelihood=internal_logit,
                prior=prior,
                observation=obs
            )

            print(
                f"[FACT UPDATE] "
                f"fact={fact:<30} "
                f"pred={predicate:<12} "
                f"prior={prior:>6.3f} "
                f"logit={internal_logit:>6.3f} "
                f"obs={str(obs):>5} "
                f"posterior={posterior:>6.3f}"
            )

            log["queries"].append({
                "fact": fact,
                "predicate": predicate,
                "prior": prior,
                "internal_logit": internal_logit,
                "oracle_obs": obs,
                "posterior": posterior,
            })

            if (predicate not in min_post_by_pred) or (posterior < min_post_by_pred[predicate][0]):
                min_post_by_pred[predicate] = (posterior, fact)

        # 2) predicate 단위 Beta 업데이트 (min posterior only)
        print("[BETA UPDATE] pessimistic per-predicate (min posterior only)")
        updated_preds: List[str] = []

        for pred, (c_min, fact_min) in min_post_by_pred.items():
            a_new, b_new = self.update_beta_distribution(pred, c_min)
            updated_preds.append(pred)

            print(
                f"  pred={pred:<12} "
                f"picked_fact={fact_min:<30} "
                f"c_min={c_min:>6.3f} "
                f"-> alpha={a_new:>7.3f}, beta={b_new:>7.3f}"
            )

            log["beta_updates"].append({
                "predicate": pred,
                "picked_fact": fact_min,
                "c_min": c_min,
                "alpha": a_new,
                "beta": b_new,
            })

        # 3) 강한 전파로 KB 갱신 (업데이트된 predicate 전체)
        print("[KB PROPAGATION] strong propagation: overwrite all facts of updated predicates")
        for pred in updated_preds:
            theta = self.propagate_predicate_confidence(kb, pred, n_samples=self.n_sample)
            log["kb_updates"].append({
                "predicate": pred,
                "theta": theta,
            })

        return log

    def run_plan(self, plan: List[Action], kb: KnowledgeBase) -> List[Dict]:
        """ordered plan을 순서대로 action 단위 처리"""
        logs: List[Dict] = []
        for action in plan:
            print(f"\nAction {action.name}")
            logs.append(self.handle_action(action, kb))
        return logs

    def run_and_record(self, plan: List[Action], kb: KnowledgeBase):
        history = {pred: [] for pred in PREDICATES}
        self.init_beta_dist(kb)
        for t, action in enumerate(plan):
            print(f"\nAction {action.name}")
            self.handle_action(action, kb)  # 내부에서 KB 업데이트까지 수행

            # timestep 끝난 뒤 KB에서 predicate별 confidence 기록
            for pred in PREDICATES:
                history[pred].append(kb.get_predicate_confidence(pred))
        query_rate = (self.number_of_query / len(plan))*100
        return history, query_rate
