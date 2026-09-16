"""Run IntroPlan's BRL adapter using the existing domain dynamics and oracle."""
from __future__ import annotations

import argparse
from datetime import datetime
import importlib
import os
from pathlib import Path
import shlex
import subprocess
import sys

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
KNOWNO = HERE.parent / 'knowno'
sys.path.insert(0, str(KNOWNO))
from knowno_baseline_experiment import parse_args, build_command, scene_id
from policy import IntrospectiveLLM


def main():
    adapter = argparse.ArgumentParser(add_help=False)
    adapter.add_argument('--knowledge', default=str(HERE / 'knowledge.json'))
    adapter.add_argument('--top-k', type=int, default=3)
    adapter.add_argument('--worker', choices=['tomato', 'wastesorting'])
    extra, remaining = adapter.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    if extra.worker:
        # Module imports remain isolated inside this worker process.
        from scripts.llm import call_llm
        planner = importlib.import_module('scripts.knowno_multistep_' + extra.worker)
        args = planner.parse_args()
        if args.run_calibration:
            raise ValueError('Use IntroPlan compute_qhat.py for calibration, then pass --qhat.')
        policy = IntrospectiveLLM(call_llm, extra.knowledge, extra.worker,
                                  args.log_file + '.introplan.jsonl', extra.top_k)
        planner.main(call_llm=policy, baseline_name='IntroPlan-BRL')
        return
    args, passthrough = parse_args()
    domain = 'wastesorting' if args.domain == 'waste' else args.domain
    args.scene = scene_id(args.scene)
    if args.max_steps is None:
        args.max_steps = 50
    if args.run_calibration:
        raise ValueError('Use IntroPlan compute_qhat.py for calibration, then pass --qhat.')
    if args.qhat is None:
        args.qhat = {'tomato': 0.9809474992495626, 'wastesorting': 0.9615342162270937}[domain]
        print('[IntroPlan] qhat from recovered 100-record calibration, gpt-4o, T=5, coverage=0.95, legacy_higher.', flush=True)
    if not 0 <= args.qhat <= 1:
        raise ValueError('qhat must be between 0 and 1')
    if args.score_temperature is None:
        args.score_temperature = 5.0
    if not args.log_file:
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        args.log_file = str(PROJECT_ROOT / 'experiments_logs/system_log' / domain /
                            f'scene_{args.scene}_step{args.max_steps or 50}' / 'when_introplan' /
                            f'introplan_{stamp}.txt')
    cmd = build_command(args, passthrough)
    cmd[1] = str(Path(__file__).resolve())
    cmd[2:2] = ['--worker', domain, '--knowledge', str(Path(extra.knowledge).resolve()), '--top-k', str(extra.top_k)]
    print('Running:', shlex.join(cmd), flush=True)
    if args.dry_run:
        return
    Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=str(KNOWNO), PYTHONUNBUFFERED='1')
    subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=True)


if __name__ == '__main__':
    main()
