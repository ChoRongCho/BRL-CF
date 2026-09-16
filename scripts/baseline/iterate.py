"""Batch engine configured by run/iterate_baseline.sh; one episode at a time."""
from __future__ import annotations

import csv
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / 'run/run_query_baseline.sh'


def setting(name, default):
    return os.environ.get('BATCH_' + name, default)


def boolean(name, default):
    value = setting(name, default)
    if value not in ('true', 'false'):
        raise ValueError(f'{name} must be true or false')
    return value == 'true'


def absolute(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def paired_rows(path, domains, scenes, repetitions):
    """Validate the entire requested matrix before moving or running anything."""
    expected = {(d, s, i) for d in domains for s in scenes for i in range(1, repetitions + 1)}
    rows = {}
    with path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        if not {'domain', 'condition', 'scene', 'iteration', 'seed'} <= set(reader.fieldnames or []):
            raise ValueError('Seed CSV is missing required columns')
        for row in reader:
            if row['condition'] != 'ours' or row['domain'] not in domains:
                continue
            key = (row['domain'], int(row['scene']), int(row['iteration']))
            if key not in expected:
                continue
            if key in rows:
                raise ValueError(f'Duplicate paired seed row: {key}')
            seed = int(row['seed'])
            if not 0 <= seed < 2**32:
                raise ValueError(f'Invalid seed: {key}')
            rows[key] = seed
    missing = expected - rows.keys()
    if missing:
        raise ValueError(f'Missing {len(missing)} paired seeds, e.g. {sorted(missing)[:3]}')
    return [(d, s, i, rows[d, s, i]) for d in domains for s in scenes for i in range(1, repetitions + 1)]


def main():
    aliases = {k: 'query_action_pomcp' for k in ('query_action_pomdp', 'query-as-action', 'targeted_query_pomdp')}
    baselines = [aliases.get(b, b) for b in setting('BASELINES', 'knowno introplan query_action_pomcp').split()]
    folders = dict(knowno='when_knowno_gpt4', introplan='when_introplan', query_action_pomcp='query_as_action')
    domains = setting('DOMAINS', 'tomato wastesorting').split()
    scenes = [int(s) for s in setting('SCENES', '1 2 3 4 5').split()]
    repetitions, steps = int(setting('ITERATIONS', '40')), int(setting('MAX_STEPS', '50'))
    for values in (baselines, domains, scenes):
        if not values or len(values) != len(set(values)):
            raise ValueError('Selections must be nonempty and contain no duplicates')
    if set(baselines) - folders.keys() or set(domains) - {'tomato', 'wastesorting'}:
        raise ValueError('Unknown baseline or domain')
    if min(scenes + [repetitions, steps]) < 1:
        raise ValueError('Scene, repetition and step counts must be positive')
    dry_run, resume = boolean('DRY_RUN', 'false'), boolean('RESUME', 'false')
    archive = boolean('ARCHIVE_EXISTING', 'true') and not resume
    seed_path = absolute(setting('PAIRED_SEED_LOG', 'experiments_logs/system_log/when_what_seed_logs/iterate_when_what_20260912_150132.csv'))
    log_root = absolute(setting('LOG_ROOT', 'experiments_logs/system_log'))
    rows = paired_rows(seed_path, domains, scenes, repetitions)
    for domain in domains:
        for scene in scenes:
            for filename in (f'scene_{scene:02}.yaml', 'domain_rule.yaml', 'robot_skill.yaml'):
                if not (ROOT / 'scripts/domain' / domain / filename).is_file():
                    raise ValueError(f'Missing domain file: {domain}/{filename}')
    configs = {}
    qhats = {'knowno': {'tomato': setting('KNOWNO_TOMATO_QHAT', '.8404'), 'wastesorting': setting('KNOWNO_WASTE_QHAT', '.8704')},
             'introplan': {'tomato': setting('INTROPLAN_TOMATO_QHAT', '.9809474992495626'), 'wastesorting': setting('INTROPLAN_WASTE_QHAT', '.9615342162270937')}}
    query_defaults = dict(N_SIMULATIONS='100', MAX_DEPTH='20', GAMMA='.95', UCB_C='1.0', EPSILON='.005',
                          MAX_PARTICLES='250', MAX_BELIEF_PARTICLES='8000', MAX_NODE_PARTICLES='8000',
                          QUERY_COST='1.0', FAILURE_PENALTY='10.0', ANSWER_ACCURACY='1.0', MAX_CONSECUTIVE_QUERIES='30')
    for baseline in baselines:
        config = {'MAX_STEPS': str(steps), 'MAX_STEP': str(steps)}
        if baseline == 'query_action_pomcp':
            config.update({k: setting(k, v) for k, v in query_defaults.items()})
            integer_keys = ('N_SIMULATIONS', 'MAX_DEPTH', 'MAX_PARTICLES', 'MAX_BELIEF_PARTICLES', 'MAX_NODE_PARTICLES', 'MAX_CONSECUTIVE_QUERIES')
            for k in integer_keys:
                if int(config[k]) < 1: raise ValueError(f'{k} must be positive')
            for k in set(query_defaults) - set(integer_keys):
                if not math.isfinite(float(config[k])) or float(config[k]) < 0: raise ValueError(f'Invalid {k}')
            if not 0 < float(config['GAMMA']) <= 1 or not 0 <= float(config['ANSWER_ACCURACY']) <= 1:
                raise ValueError('Invalid gamma or answer accuracy')
        else:
            config.update(PROMPT_VERSION=setting('PROMPT_VERSION', 'v2'), SCORE_TEMPERATURE=setting('SCORE_TEMPERATURE', '5.0'))
            if config['PROMPT_VERSION'] not in ('v1', 'v2') or not math.isfinite(float(config['SCORE_TEMPERATURE'])) or float(config['SCORE_TEMPERATURE']) <= 0:
                raise ValueError('Invalid prompt version or score temperature')
            if any(not 0 <= float(q) <= 1 for q in qhats[baseline].values()): raise ValueError('Invalid qhat')
            config['settings_sha256'] = hashlib.sha256((ROOT / 'llm_setting.json').read_bytes()).hexdigest()
            if baseline == 'introplan':
                knowledge = absolute(setting('KNOWLEDGE_FILE', 'scripts/baseline/introplan/knowledge.json'))
                data = knowledge.read_bytes()
                if any(not json.loads(data).get(d) for d in domains): raise ValueError('Missing domain knowledge')
                config.update(TOP_K=setting('TOP_K', '3'), KNOWLEDGE_FILE=str(knowledge), knowledge_sha256=hashlib.sha256(data).hexdigest())
                if int(config['TOP_K']) < 1: raise ValueError('TOP_K must be positive')
        configs[baseline] = config
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    plans = [(b, *row) for b in baselines for row in rows]
    print(f'Baselines: {baselines}\nDomains: {domains}; scenes: {scenes}\nRepetitions per baseline/domain/scene: {repetitions}\nTotal episodes: {len(plans)}', flush=True)
    def episode_dir(b, d, s):
        return log_root / d / f'scene_{s:02}_step{steps}' / folders[b]
    if archive and not dry_run:
        for b in baselines:
            for d in domains:
                for s in scenes:
                    path = episode_dir(b, d, s)
                    if path.exists():
                        destination = log_root / 'archive' / f'baselines_{timestamp}' / d / path.parent.name / path.name
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(path), str(destination))
    seed_log = log_root / 'baseline_seed_logs' / f'iterate_baseline_{timestamp}.csv'
    fields = ['global_index', 'baseline', 'domain', 'scene', 'iteration', 'seed', 'max_steps', 'prompt_version', 'qhat', 'temperature',
              'top_k', 'n_simulations', 'max_depth', 'query_cost', 'failure_penalty', 'answer_accuracy', 'expert', 'status', 'log_path']
    if not dry_run:
        seed_log.parent.mkdir(parents=True, exist_ok=True)
        with seed_log.open('w', newline='') as f: csv.writer(f).writerow(fields)
        seed_log.with_suffix('.json').write_text(json.dumps(dict(configs=configs, qhats=qhats, baselines=baselines,
            domains=domains, scenes=scenes, repetitions=repetitions, paired_seed_log=str(seed_path),
            paired_seed_sha256=hashlib.sha256(seed_path.read_bytes()).hexdigest()), indent=2))
    failed = skipped = 0
    for index, (baseline, domain, scene, iteration, seed) in enumerate(plans, 1):
        config = dict(configs[baseline])
        if baseline in qhats: config['QHAT'] = qhats[baseline][domain]
        fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        log_dir = episode_dir(baseline, domain, scene)
        stem = f'{baseline}_{domain}_scene{scene:02}_iter{iteration:02}_seed{seed}'
        log_path = log_dir if baseline == 'query_action_pomcp' else log_dir / (stem + '.txt')
        marker = log_dir / '.completed' / (stem + '.done')
        env = dict(os.environ, **{k:v for k,v in config.items() if k.isupper()}, BASELINE=baseline, DOMAIN=domain,
                   SCENE=f'{scene:02}', SEED=str(seed), LOG_ROOT=str(log_root), LOG_FILE=str(log_path),
                   AUTO_ANSWER='true', DRY_RUN='true' if dry_run else 'false', PYTHONUNBUFFERED='1')
        if dry_run:
            if scene == scenes[0] and iteration == 1:
                print(f'\n{baseline}/{domain}: {shlex.quote(str(log_path))}', flush=True)
                subprocess.run(['bash', str(RUNNER)], cwd=ROOT, env=env, check=True)
            continue
        status = 'complete'
        saved = marker.read_text().strip() if marker.exists() else None
        # Older KnowNo/QaA runners wrote empty completion markers without settings metadata.
        legacy_complete = saved == '' and baseline != 'introplan' and log_path.exists()
        if baseline == 'introplan' and saved:
            old_config = dict(domains=domains, scenes=scenes, max_steps=steps,
                              qhats={d: float(q) for d, q in qhats['introplan'].items()},
                              temperature=float(config['SCORE_TEMPERATURE']), prompt_version=config['PROMPT_VERSION'],
                              top_k=int(config['TOP_K']), knowledge_sha256=config['knowledge_sha256'],
                              settings_sha256=config['settings_sha256'])
            legacy_complete = saved == hashlib.sha256(json.dumps(old_config, sort_keys=True).encode()).hexdigest() and log_path.exists()
        if resume and (saved == fingerprint or legacy_complete):
            status = 'skipped'; skipped += 1
        else:
            log_dir.mkdir(parents=True, exist_ok=True); marker.parent.mkdir(exist_ok=True)
            marker.unlink(missing_ok=True)
            if baseline == 'introplan': Path(str(log_path) + '.introplan.jsonl').unlink(missing_ok=True)
            with (log_dir / (stem + '.console.log')).open('w') as stream:
                result = subprocess.run(['bash', str(RUNNER)], cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
            if result.returncode == 0: marker.write_text(fingerprint + '\n')
            else: status = 'failed'; failed += 1
        with seed_log.open('a', newline='') as f:
            csv.writer(f).writerow([index, baseline, domain, f'{scene:02}', iteration, seed, steps,
                config.get('PROMPT_VERSION',''), config.get('QHAT',''), config.get('SCORE_TEMPERATURE',''), config.get('TOP_K',''),
                config.get('N_SIMULATIONS',''), config.get('MAX_DEPTH',''), config.get('QUERY_COST',''),
                config.get('FAILURE_PENALTY',''), config.get('ANSWER_ACCURACY',''), 'exact_oracle', status, str(log_path)])
        print(f'\rProgress: {index}/{len(plans)} {baseline} ({status})', end='', flush=True)
    print(f'\nDry run checked {len(plans)} paired runs; no API calls or file changes.' if dry_run else
          f'\nCompleted: {len(plans)-skipped-failed}, skipped: {skipped}, failed: {failed}\nSeed log: {seed_log}')
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
