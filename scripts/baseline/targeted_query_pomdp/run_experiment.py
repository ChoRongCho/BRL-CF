"""Query-as-Action POMDP baseline의 단일 episode 실행 흐름.

이 baseline은 로봇의 물리 행동과 Boolean 상태 질문을 하나의 action
공간에 넣고, POMCP가 두 종류의 행동을 같은 누적 보상 기준으로 선택하게 한다.

Pseudo code::

    1. 실행 인자와 Query-as-Action 전용 설정을 읽는다.
       - query_cost: 질문 한 번의 비용
       - failure_penalty: 상태 가설에서 실행 불가능한 물리 행동의 penalty
       - answer_accuracy: oracle 답변을 그대로 사용할 확률

    2. Environment, BeliefManager, QueryAsActionPOMCPPlanner를 생성한다.
       초기 환경 상태로 symbolic belief를 초기화한다.

    3. episode가 종료될 때까지 다음 결정을 반복한다.

       action = planner.search(belief)

       if action이 없으면:
           PLAN FAILURE로 종료한다.

       if action이 QueryAction이면:
           a. 질문 action에 지정된 target fact의 참/거짓을 oracle에 묻는다.
           b. answer_accuracy에 따라 oracle 답변 또는 반대 답변을 사용한다.
           c. 답변과 일치하는 belief particle만 남기고 확률을 정규화한다.
           d. posterior의 MAP state를 다음 planning의 symbolic knowledge로 쓴다.
           e. cumulative reward에서 query_cost를 차감한다.
           f. 질문 횟수와 질문 전후 confidence를 기록한다.

       else action이 물리 행동이면:
           a. env.step(action)으로 transition과 observation을 실행한다.
           b. 실행 reward를 cumulative reward에 더한다.
           c. action과 observation으로 belief를 갱신한다.
           d. posterior의 MAP state를 다음 planning의 symbolic knowledge로 쓴다.
           e. 물리 step, 실행 시간, observation을 기록한다.

       env.check_done(belief)가 GOAL DONE 또는 PLAN FAILURE를 반환하거나,
       물리 행동과 QueryAction을 합친 전체 action 수가 max_step에 도달하면
       episode를 종료한다.

    4. 성공 여부, 종료 이유, 물리 행동 수, 질문 수, 누적 reward, timing,
       전체 decision 순서와 최종 belief knowledge를 결과 로그로 저장한다.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys
from time import time

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
for path in (PROJECT_ROOT, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.baseline.targeted_query_pomdp.environment import Environment
from main import build_action_schema_summary, print_step_timing
from scripts.baseline.targeted_query_pomdp.belief_update import BeliefManager
from scripts.baseline.targeted_query_pomdp.planner import QueryAsActionPOMCPPlanner
from scripts.baseline.targeted_query_pomdp.query_actions import QueryAction
from utils.arguments import parse_args
from utils.logger import logger_exp


def query_action_oracle_facts(env) -> set[str]:
    """Combine hidden physical facts with QaA's execution-history facts."""
    facts = set(env.true_state.facts)
    epistemic_prefixes = ("observed(", "scanned(", "detected(")
    facts.update(
        fact for fact in env.state.facts
        if fact.startswith(epistemic_prefixes)
    )
    return facts


def parse_baseline_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--query-cost", type=float, default=1.0)
    parser.add_argument("--failure-penalty", type=float, default=10.0)
    parser.add_argument("--answer-accuracy", type=float, default=1.0)
    baseline, remaining = parser.parse_known_args()
    original = sys.argv
    try:
        sys.argv = [original[0], *remaining]
        args = parse_args("tomato")
    finally:
        sys.argv = original
    if args.answer_type != "auto":
        parser.error("This baseline currently uses the existing Boolean auto Oracle")
    if baseline.query_cost < 0.0:
        parser.error("--query-cost must be non-negative")
    if baseline.failure_penalty < 0.0:
        parser.error("--failure-penalty must be non-negative")
    args.query_as_action = baseline
    return args


def main() -> None:
    args = parse_baseline_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    env = Environment(args)
    belief_manager = BeliefManager(
        args,
        env.transition_model,
        env.observation_model,
        env.asp_bridge,
    )
    planner = QueryAsActionPOMCPPlanner(
        args,
        env,
        belief_manager,
        query_cost=args.query_as_action.query_cost,
        answer_accuracy=args.query_as_action.answer_accuracy,
        failure_penalty=args.query_as_action.failure_penalty,
    )
    env.reset()
    belief = belief_manager.initialize_belief(env.state)
    manager = belief_manager.feedback_manager

    step_logs = []
    decision_log = []
    totals = {key: 0.0 for key in (
        "search_time", "execute_time", "update_time",
        "interaction_time", "pruning_time", "step_total_time",
    )}
    cumulative_reward = 0.0
    plan_success = False
    end_reason = None
    wall_start = time()
    physical_step = 0
    decision_index = 0

    last_task_action = None
    last_observation_facts = []
    oracle_successor_facts = None

    while True:
        if decision_index >= args.max_step:
            end_reason = "MAX STEP"
            print(
                "[Planner] MAX STEP "
                f"({decision_index} total actions: "
                f"{physical_step} physical, {manager.num_of_query} query)"
            )
            break
        decision_start = time()
        phase_start = time()
        action = planner.search(belief)
        search_time = time() - phase_start
        totals["search_time"] += search_time
        if action is None:
            end_reason = "PLAN FAILURE"
            print("[Planner] PLAN FAILURE")
            break

        decision_index += 1
        decision_log.append(action.name)
        print(f"[Decision {decision_index}] Selected action: {action.name}")

        if isinstance(action, QueryAction):

            confidence_before = manager.compute_confidence(belief.frontier_weights)
            phase_start = time()
            oracle_action_name = (
                last_task_action.name if last_task_action is not None else None
            )
            oracle_answer = bool(manager.call_feedback(
                action.target_fact,
                # Reuse the same exact oracle path as Ours.  The most recent
                # physical action determines detect/scan visibility semantics;
                # the QueryAction itself is only the decision to ask.
                oracle_action_name,
                observation_facts=last_observation_facts,
                oracle_state_facts=query_action_oracle_facts(env),
                oracle_successor_facts=oracle_successor_facts,
            ))
            answer = (
                oracle_answer
                if random.random() <= args.query_as_action.answer_accuracy
                else not oracle_answer
            )
            interaction_time = time() - phase_start
            totals["interaction_time"] += interaction_time

            before_count = len(belief.frontier)
            belief = manager.apply_fact_answer_to_belief(
                belief,
                action.target_fact,
                bool(answer),
            )
            # The posterior remains a distribution. Its MAP state is the
            # symbolic KB used for applicability; the query-confirmed fact (or
            # its Boolean complement encoded by absence) is therefore promoted.
            belief.sync_knowledge_to_map()
            confidence_after = manager.compute_confidence(belief.frontier_weights)
            manager.num_of_query += 1
            manager.query_log.append({
                "step": physical_step,
                "decision": decision_index,
                "action": (
                    last_task_action.name if last_task_action is not None else None
                ),
                "query_action": action.name,
                "question": action.target_fact,
                "answer": bool(answer),
                "oracle_answer": oracle_answer,
                "answer_type": "boolean",
                "observation": list(last_observation_facts),
                "confidence_before": confidence_before,
                "confidence_after": confidence_after,
                "baseline": "Query-as-Action (QaA)",
            })
            cumulative_reward -= action.cost
            print(
                f"    [Boolean Query] {action.name} -> {bool(answer)}; "
                f"belief {before_count}->{len(belief.frontier)}, "
                f"confidence {confidence_before:.4f}->{confidence_after:.4f}"
            )
            totals["step_total_time"] += time() - decision_start

            done = env.check_done(belief)
            if done in {"GOAL DONE", "PLAN FAILURE"}:
                end_reason = done
                plan_success = done == "GOAL DONE"
                break
            continue

        physical_step += 1
        step_log = {
            "step": physical_step,
            "decision": decision_index,
            "action": action.name,
            "search_time": search_time,
            "execute_time": 0.0,
            "update_time": 0.0,
            "interaction_time": 0.0,
            "pruning_time": 0.0,
            "step_total_time": 0.0,
        }

        phase_start = time()
        observation, reward, _, _ = env.step(action)
        execute_time = time() - phase_start
        totals["execute_time"] += execute_time
        cumulative_reward += reward
        step_log["execute_time"] = execute_time
        step_log["step_reward"] = reward
        step_log["cumulated_reward"] = cumulative_reward
        print("[Planner] Observation:", observation.state.facts)

        phase_start = time()
        belief = belief_manager.update_belief(belief, observation, action)
        # Keep uncertainty for future query actions while maintaining the same
        # MAP knowledge convention used by the original BRL action planner.
        belief.sync_knowledge_to_map()
        update_time = time() - phase_start
        totals["update_time"] += update_time
        step_log["update_time"] = update_time

        last_task_action = action
        last_observation_facts = list(observation.state.facts)
        oracle_successor_facts = query_action_oracle_facts(env)

        done = env.check_done(belief)
        if done in {"GOAL DONE", "MAX STEP", "PLAN FAILURE"}:
            end_reason = done
            plan_success = done == "GOAL DONE"

        step_time = time() - decision_start
        totals["step_total_time"] += step_time
        step_log["step_total_time"] = step_time
        step_logs.append(step_log)
        print_step_timing(step_log)
        if end_reason is not None:
            break

    if end_reason is None:
        end_reason = "PLAN FAILURE"
    elapsed = time() - wall_start
    count = max(decision_index, 1)
    timing = {
        key: {"total": value, "avg": value / count}
        for key, value in totals.items()
    }
    timing["total_time"] = elapsed
    result = {
        "meta": {
            "experiment": "query_as_action",
            "policy": "Query-as-Action (QaA)",
            "query_interface": "enumerated_grounded_boolean_actions",
            "domain": args.domain,
            "initial_state": args.initial_state,
            "seed": args.seed,
            "max_step": args.max_step,
            "n_simulations": args.n_simulations,
            "max_depth": args.max_depth,
            "gamma": args.gamma,
            "c": args.c,
            "epsilon": args.epsilon,
            "max_particles": args.max_particles,
            "max_belief_particles": args.max_belief_particles,
            "max_node_particles": planner.max_node_particles,
            "query_cost": args.query_as_action.query_cost,
            "failure_penalty": args.query_as_action.failure_penalty,
            "answer_accuracy": args.query_as_action.answer_accuracy,
            "log_dir": args.log_dir,
        },
        "success": plan_success,
        "end_reason": end_reason,
        "steps": decision_index,
        "physical_steps": len(step_logs),
        "total_actions": decision_index,
        "reward": {"cumulated": cumulative_reward},
        "actions": step_logs,
        "decisions": decision_log,
        "timing": timing,
        "questions": manager.query_log,
        "total_questions": manager.num_of_query,
        "action_schema_summary": build_action_schema_summary(
            step_logs,
            manager.query_log,
        ),
        "final_knowledge": {
            "facts": sorted(belief.knowledge.facts),
            "fluents": belief.knowledge.fluents,
        },
    }
    log_path = logger_exp(result, log_dir=args.log_dir)
    print("=============Query-as-Action (QaA)=============")
    print(f"Success: {plan_success} ({end_reason})")
    print(
        f"Total actions: {decision_index}, Physical steps: {len(step_logs)}, "
        f"Total Query: {manager.num_of_query}"
    )
    print(f"Log: {log_path}")


if __name__ == "__main__":
    main()
