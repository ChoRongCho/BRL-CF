"""
Domain smoke test.

Usage examples:
  python3 test.py
  python3 test.py --domains watering rover kitchen
  python3 test.py --run random --horizon 20
  python3 test.py --run pomcp --horizon 20
  python3 test.py --horizon 20 --keep-going

Arguments:
  --domains
      테스트할 도메인 이름 목록이다. 생략하면 tomato, wastesorting,
      blocksworld, kitchen, rover, watering을 모두 테스트한다.
      예: --domains watering rover

  --seed
      랜덤 액션 선택, stochastic transition/observation sampling에 쓰는 seed다.
      같은 seed를 쓰면 가능한 한 같은 랜덤 실행을 재현할 수 있다.

  --horizon
      각 도메인에서 action을 최대 몇 번 실행할지 정한다.
      기본값은 20이다. goal을 달성하거나 applicable action이 없으면 그 전에 멈춘다.

  --run
      action 선택 방식을 정한다.
      random: 현재 state에서 applicable action을 무작위로 선택한다.
      pomcp: POMCPPlanner.search()로 action을 선택한다.

  --keep-going
      어떤 도메인이 실패해도 다음 도메인 테스트를 계속 진행한다.
      이 옵션이 없으면 첫 실패 지점에서 바로 종료한다.

Output:
  test_result.md
      test.py를 실행할 때마다 새로 덮어쓰는 markdown log 파일이다.
      도메인별 action plan, observation, reward, state diff를 기록한다.

This file creates Environment directly. In random mode it samples applicable
actions directly; in pomcp mode it uses POMCPPlanner.search() to choose actions.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

try:
    import numpy as np
except ModuleNotFoundError:
    np = None


PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
RESULT_PATH = PROJECT_ROOT / "test_result.md"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

DEFAULT_DOMAINS = (
    "tomato",
    "wastesorting",
    "blocksworld",
    "kitchen",
    "rover",
    "watering",
)

PLAN_TABLE_DOMAINS = (
    "tomato",
    "wastesorting",
    "blocksworld",
    "rover",
    "kitchen",
    "watering",
)


def make_args(domain: str, seed: int, horizon: int) -> SimpleNamespace:
    domain_root = SCRIPTS_DIR / "domain" / domain
    return SimpleNamespace(
        domain=domain,
        domain_rule=domain_root / "domain_rule.yaml",
        initial_state=domain_root / "scene_01.yaml",
        robot_skill=domain_root / "robot_skill.yaml",
        seed=seed,
        max_step=horizon,
        gamma=0.95,
        c=1.0,
        max_depth=20,
        n_simulations=10,
        max_node_particles=None,
        epsilon=0.005,
        max_particles=100,
        max_belief_particles=200,
        threshold=0.8,
        log_dir=PROJECT_ROOT / "experiments_logs" / "debug_test",
        fluent_sample_sigma=0.05,
        pick_fluent_sigma=0.05,
        pick_success_rate=0.88,
        pick_success_floor=0.05,
        f_strategy=1,
        q_strategy=1,
        answer_type="oracle",
        random_query_prob=0.0,
    )


def format_facts(facts: Iterable[str], limit: int = 8) -> str:
    facts = list(facts)
    shown = ", ".join(facts[:limit])
    if len(facts) > limit:
        shown += f", ... (+{len(facts) - limit})"
    return shown or "-"


def format_fact_block(facts: Iterable[str]) -> str:
    facts = sorted(facts)
    return "\n".join(f"- `{fact}`" for fact in facts) or "- `-`"


def escape_table_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def goal_achieved(env) -> bool:
    return all(env.state.has_fact(fact) for fact in env.goal.facts)


def run_random_episode(env, horizon: int, rng: random.Random) -> dict[str, Any]:
    episode_log: dict[str, Any] = {
        "horizon": horizon,
        "steps": [],
        "end_reason": "horizon <= 0",
        "cumulative_reward": 0.0,
    }
    if horizon <= 0:
        return episode_log

    cumulative_reward = 0.0
    print(f"  [episode] random applicable actions, horizon={horizon}, no POMCP")
    for step in range(1, horizon + 1):
        applicable = [action for action in env.actions if action.is_applicable(env.state)]
        if not applicable:
            print(f"    step {step}: no applicable actions")
            episode_log["end_reason"] = "no applicable actions"
            break

        action = rng.choice(applicable)
        before = set(env.state.facts)
        observation, reward, done, info = env.step(action)
        cumulative_reward += reward
        after = set(env.state.facts)
        added = sorted(after - before)
        removed = sorted(before - after)
        step_log = {
            "step": step,
            "action": action.name,
            "observation": sorted(observation.state.facts),
            "reward": reward,
            "cumulative_reward": cumulative_reward,
            "added": added,
            "removed": removed,
            "applicable_next": len(info.get("applicable_actions", [])),
        }
        episode_log["steps"].append(step_log)

        print(f"    step {step:02d}: {action.name}")
        print(f"      obs: {format_facts(observation.state.facts)}")
        print(f"      reward: {reward} / cumulative: {cumulative_reward}")
        print(f"      added: {format_facts(added)}")
        print(f"      removed: {format_facts(removed)}")
        print(f"      applicable_next: {len(info.get('applicable_actions', []))}")

        if done:
            print(f"      env done: {done}")
            episode_log["end_reason"] = f"env done: {done}"
            break
        if goal_achieved(env):
            print("      goal achieved")
            episode_log["end_reason"] = "goal achieved"
            break
    else:
        episode_log["end_reason"] = "horizon reached"

    episode_log["cumulative_reward"] = cumulative_reward
    return episode_log


def log_executed_step(
    step: int,
    action,
    observation,
    reward: float,
    cumulative_reward: float,
    before: set[str],
    after: set[str],
    info: dict[str, Any],
) -> dict[str, Any]:
    added = sorted(after - before)
    removed = sorted(before - after)
    step_log = {
        "step": step,
        "action": action.name,
        "observation": sorted(observation.state.facts),
        "reward": reward,
        "cumulative_reward": cumulative_reward,
        "added": added,
        "removed": removed,
        "applicable_next": len(info.get("applicable_actions", [])),
    }

    print(f"    step {step:02d}: {action.name}")
    print(f"      obs: {format_facts(observation.state.facts)}")
    print(f"      reward: {reward} / cumulative: {cumulative_reward}")
    print(f"      added: {format_facts(added)}")
    print(f"      removed: {format_facts(removed)}")
    print(f"      applicable_next: {len(info.get('applicable_actions', []))}")

    return step_log


def run_pomcp_episode(env, args: SimpleNamespace, horizon: int) -> dict[str, Any]:
    episode_log: dict[str, Any] = {
        "horizon": horizon,
        "steps": [],
        "end_reason": "horizon <= 0",
        "cumulative_reward": 0.0,
    }
    if horizon <= 0:
        return episode_log

    try:
        from models.belief_update import BeliefManager
        from planners.pomcp import POMCPPlanner
    except ModuleNotFoundError as exc:
        missing = exc.name or str(exc)
        raise RuntimeError(
            f"missing runtime dependency '{missing}'. "
            f"Install dependencies with: python3 -m pip install -r requirements.txt"
        ) from exc

    belief_manager = BeliefManager(
        args,
        env.transition_model,
        env.observation_model,
        env.asp_bridge,
    )
    planner = POMCPPlanner(args=args, env=env, belief_manager=belief_manager)
    belief = belief_manager.initialize_belief(env.state)

    cumulative_reward = 0.0
    print(f"  [episode] POMCP actions, horizon={horizon}")
    for step in range(1, horizon + 1):
        action = planner.search(belief)
        if action is None:
            print(f"    step {step}: planner returned no action")
            episode_log["end_reason"] = "planner returned no action"
            break

        before = set(env.state.facts)
        observation, reward, done, info = env.step(action)
        cumulative_reward += reward
        after = set(env.state.facts)

        episode_log["steps"].append(
            log_executed_step(
                step=step,
                action=action,
                observation=observation,
                reward=reward,
                cumulative_reward=cumulative_reward,
                before=before,
                after=after,
                info=info,
            )
        )

        belief = belief_manager.update_belief(belief, observation, action)
        belief = belief_manager.feedback_manager.get_new_observation(
            belief=belief,
            step=step,
            action_name=action.name,
        )
        if done:
            print(f"      env done: {done}")
            episode_log["end_reason"] = f"env done: {done}"
            break
        done_reason = env.check_done(belief=belief)
        if done_reason:
            episode_log["end_reason"] = str(done_reason)
            break

        planner.prune_search_tree(action=action, obs=belief.knowledge)
    else:
        episode_log["end_reason"] = "horizon reached"

    episode_log["cumulative_reward"] = cumulative_reward
    return episode_log


def test_domain(domain: str, seed: int, horizon: int, run_mode: str) -> dict[str, Any]:
    try:
        from environments.env import Environment
    except ModuleNotFoundError as exc:
        missing = exc.name or str(exc)
        raise RuntimeError(
            f"missing runtime dependency '{missing}'. "
            f"Install dependencies with: python3 -m pip install -r requirements.txt"
        ) from exc

    print(f"\n=== {domain} ===")
    args = make_args(domain, seed=seed, horizon=max(horizon, 1))
    domain_log: dict[str, Any] = {
        "domain": domain,
        "status": "ok",
        "run_mode": run_mode,
        "seed": seed,
        "horizon": horizon,
        "initial_state_path": str(args.initial_state),
        "domain_rule_path": str(args.domain_rule),
        "robot_skill_path": str(args.robot_skill),
    }

    rng = random.Random(seed)
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)

    env = Environment(args)
    env.reset()

    print(f"  state facts: {len(env.state.facts)}")
    print(f"  true facts: {len(env.true_state.facts)}")
    print(f"  goal facts: {len(env.goal.facts)}")
    print(f"  grounded actions: {len(env.actions)}")
    domain_log.update({
        "state_fact_count": len(env.state.facts),
        "true_fact_count": len(env.true_state.facts),
        "goal_fact_count": len(env.goal.facts),
        "grounded_action_count": len(env.actions),
        "initial_state": sorted(env.state.facts),
        "goal": sorted(env.goal.facts),
    })
    if not env.actions:
        raise AssertionError(f"{domain} generated no grounded actions")

    if run_mode == "random":
        domain_log["episode"] = run_random_episode(env, horizon, rng)
    elif run_mode == "pomcp":
        domain_log["episode"] = run_pomcp_episode(env, args, horizon)
    else:
        raise ValueError(f"unknown run mode: {run_mode}")

    return domain_log


def write_test_result(
    results: list[dict[str, Any]],
    failures: list[tuple[str, str]],
    seed: int,
    horizon: int,
    domains: list[str],
    run_mode: str,
) -> None:
    result_by_domain = {result["domain"]: result for result in results}
    lines = [
        "# Domain Test Result",
        "",
        "이 파일은 `python3 test.py` 실행 시마다 새로 덮어쓴다.",
        "",
        "## Summary",
        "",
        f"- Domains: {', '.join(domains)}",
        f"- Run mode: `{run_mode}`",
        f"- Seed: `{seed}`",
        f"- Horizon: `{horizon}`",
        f"- POMCP: {'used' if run_mode == 'pomcp' else 'not used'}",
        f"- Passed: `{len(results)}`",
        f"- Failed: `{len(failures)}`",
        "",
    ]

    lines.extend(build_plan_table(result_by_domain, horizon))

    if failures:
        lines.extend(["## Failures", ""])
        for domain, message in failures:
            lines.extend([f"### {domain}", "", f"- Error: `{message}`", ""])

    for result in results:
        episode = result.get("episode", {})
        lines.extend([
            f"## {result['domain']}",
            "",
            f"- Status: `{result['status']}`",
            f"- Initial state: `{result['initial_state_path']}`",
            f"- Domain rule: `{result['domain_rule_path']}`",
            f"- Robot skill: `{result['robot_skill_path']}`",
            f"- State facts: `{result['state_fact_count']}`",
            f"- True facts: `{result['true_fact_count']}`",
            f"- Goal facts: `{result['goal_fact_count']}`",
            f"- Grounded actions: `{result['grounded_action_count']}`",
            f"- End reason: `{episode.get('end_reason', '-')}`",
            f"- Cumulative reward: `{episode.get('cumulative_reward', 0.0)}`",
            "",
            "### Goal",
            "",
            format_fact_block(result.get("goal", [])),
            "",
            f"### {run_mode.upper()} Plan",
            "",
        ])

        steps = episode.get("steps", [])
        if not steps:
            lines.extend(["- No actions executed.", ""])
            continue

        for step in steps:
            lines.extend([
                f"#### Step {step['step']:02d}",
                "",
                f"- Action: `{step['action']}`",
                f"- Reward: `{step['reward']}`",
                f"- Cumulative reward: `{step['cumulative_reward']}`",
                f"- Applicable actions after step: `{step['applicable_next']}`",
                "",
                "Observation:",
                "",
                format_fact_block(step["observation"]),
                "",
                "Added facts:",
                "",
                format_fact_block(step["added"]),
                "",
                "Removed facts:",
                "",
                format_fact_block(step["removed"]),
                "",
            ])

    RESULT_PATH.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def build_plan_table(result_by_domain: dict[str, dict[str, Any]], horizon: int) -> list[str]:
    headers = ["", *PLAN_TABLE_DOMAINS]
    lines = [
        "## Plan Table",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]

    for step_number in range(1, horizon + 1):
        row = [f"plan{step_number}"]
        for domain in PLAN_TABLE_DOMAINS:
            result = result_by_domain.get(domain)
            steps = result.get("episode", {}).get("steps", []) if result else []
            action = "-"
            for step in steps:
                if step.get("step") == step_number:
                    action = step.get("action", "-")
                    break
            row.append(f"`{escape_table_cell(action)}`" if action != "-" else "-")

        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Debug smoke test for symbolic domains using random actions or POMCP."
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        default=list(DEFAULT_DOMAINS),
        help="Domain names to test.",
    )
    parser.add_argument(
        "--run",
        choices=("random", "pomcp"),
        default="random",
        help="Action selection mode. random samples applicable actions; pomcp uses POMCPPlanner.search().",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed for action/transition/observation sampling.")
    parser.add_argument("--horizon", type=int, default=20, help="Episode horizon per domain.")
    parser.add_argument("--keep-going", action="store_true", help="Continue testing after a domain fails.")
    args = parser.parse_args()

    results: list[dict[str, Any]] = []
    failures: list[tuple[str, str]] = []
    for domain in args.domains:
        try:
            results.append(test_domain(domain, seed=args.seed, horizon=args.horizon, run_mode=args.run))
            print(f"  [ok] {domain}")
        except Exception as exc:
            failures.append((domain, str(exc)))
            print(f"  [fail] {domain}: {exc}")
            if not args.keep_going:
                break

    write_test_result(
        results=results,
        failures=failures,
        seed=args.seed,
        horizon=args.horizon,
        domains=args.domains,
        run_mode=args.run,
    )
    print(f"\nWrote markdown log: {RESULT_PATH}")

    if failures:
        print("\nFailures:")
        for domain, message in failures:
            print(f"  - {domain}: {message}")
        return 1

    print("\nAll requested domains passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
