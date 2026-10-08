"""Classify failures from candidate feasibility, execution history, and VLM responses."""
from pathlib import Path
import ast
import csv
import hashlib
import re
from collections import Counter

manuscript = Path(__file__).resolve().parents[1]
root = manuscript.parents[1] / 'experiments_final/02_VLM/00_vlm_exp'
with (root / '01_processed/failures.csv').open(encoding='utf-8-sig') as f:
    rows = list(csv.DictReader(f))
def context_at_step(planner, steps, step):
    """Use completed physical actions for gripper state, not stale holding facts."""
    held, location, processed = None, 'dock_station', set()
    for n, action in steps:
        if int(n) >= int(step):
            break
        if action.startswith('navigate to '):
            location = action.removeprefix('navigate to ')
        elif action.startswith('pick '):
            held = action.removeprefix('pick ')
        elif action.startswith(('place ', 'discard ')):
            processed.add(action.split()[1])
            held = None
    before = planner.split('[KnowNo] Step ' + str(step) + ' |')[0]
    facts = set()
    for line in re.findall(r'\[KnowNo\] Observation facts[^:]*: (.*)', before):
        for fact in ast.literal_eval(line):
            if fact.startswith(('fresh(', 'rotten(', 'ripe(', 'unripe(')):
                name, arg = fact.split('(', 1)
                opposite = {'fresh': 'rotten', 'rotten': 'fresh', 'ripe': 'unripe', 'unripe': 'ripe'}[name] + '(' + arg
                facts.discard(opposite)
            facts.add(fact)
    return held, location, processed, facts


def feasible(action, state):
    held, location, processed, facts = state
    if action.startswith('navigate to '):
        return held is None
    if action.startswith('detect '):
        return held is None and action.split()[1] == location
    if action.startswith('pick '):
        tomato = action.split()[1]
        return (held is None and tomato not in processed and
                f'observed({tomato})' in facts and f'ripe({tomato})' in facts and
                f'at({tomato},{location})' in facts)
    if action.startswith('scan '):
        return held == action.split()[1]
    if action.startswith(('place ', 'discard ')):
        verb, tomato = action.split()[:2]
        kind = 'fresh' if verb == 'place' else 'rotten'
        return held == tomato and f'scanned({tomato})' in facts and f'{kind}({tomato})' in facts
    return False


classified = []
for r in rows:
    path = root / r['raw_copy']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == r['raw_sha256']
    text = path.read_text()
    planner = (path.parent.parent / 'planner.log').read_text()
    cat = r['failure_category']
    steps = re.findall(r'^STEP (\d+): (.*?)(?: \(search=)', text, re.M)
    section = text.split('[Questions]')[1].split('[Final Knowledge]')[0]
    query_ids = set(re.findall(r'^Step=(\d+):', section, re.M))
    terminal_queried = bool(steps and steps[-1][0] in query_ids)
    offered, valid, evidence = [], [], ''
    blocks = list(re.finditer(r'^Step=(\d+):\n(.*?)(?=^Step=|\Z)', section, re.M | re.S))
    if r['method'] == 'knowno' and terminal_queried:
        block = next(m for m in blocks if m[1] == steps[-1][0])
        offered = ast.literal_eval(re.search(r'- Observation: (.*)', block[2])[1])
        state = context_at_step(planner, steps, steps[-1][0])
        valid = [a for a in offered if feasible(a.split('=', 1)[1], state)]
        evidence = f'Before terminal action held={state[0]} location={state[1]} processed={sorted(state[2])}'
    if cat == 'vlm_candidate_token_parse_failure':
        group, basis = 'VLM response', 'VLM output cannot be parsed as a candidate token'
    elif cat == 'precondition_failure':
        sets = re.findall(r'\[KnowNo\] prediction set: (.*)', planner)
        assert len(ast.literal_eval(sets[-1])) == 1
        group, basis = 'Planner decision', 'Singleton prediction set selects an action with an unmet precondition without a query'
    elif cat == 'max steps reached':
        # Review the trajectory, not the max-step ending alone.
        inefficient = []
        for block in blocks:
            options = ast.literal_eval(re.search(r'- Observation: (.*)', block[2])[1])
            answer = re.search(r'answer=(.*?), confidence=', block[2])[1]
            selected = next((a.split('=', 1)[1] for a in options if a.split('=', 1)[0] == answer), '')
            state = context_at_step(planner, steps, block[1])
            alternatives = [a for a in options if feasible(a.split('=', 1)[1], state) and a.split('=', 1)[1].startswith(('navigate to ', 'pick '))]
            if selected.startswith('detect ') and alternatives:
                inefficient.append(f'Step {block[1]} selects {selected} despite {alternatives}')
        assert inefficient, r['run']
        group, basis = 'VLM inefficient action selection', 'Repeated detection despite feasible task-progress alternatives leads to the step limit'
        evidence = ' | '.join(inefficient)

    elif cat.startswith('physical action failure:'):
        group, basis = 'VLM decision (user-confirmed)', 'The researcher confirms that this action failure label records an intervention after an incorrect VLM decision, not a mechanical execution failure'
    elif cat == 'expert_reported_failure' and r['method'] == 'knowno':
        if terminal_queried and valid:
            group, basis = 'VLM action selection', 'VLM selects the evaluator-rejected action although a feasible concrete candidate exists in the logged decision context'
        elif terminal_queried:
            group, basis = 'Planner infeasible candidate set', 'No offered concrete action satisfies preconditions in the logged decision context'
        else:
            group, basis = 'Planner autonomous action', 'Autonomous prediction-set selection ends in evaluator-reported task failure without external selection at that step'

    elif cat == 'expert_reported_failure' and r['method'] == 'pomdp':
        group, basis = 'VLM-feedback-associated task failure', 'Evaluator reports task failure after belief updates with VLM responses; VLM error is not independently established'
    else:
        raise ValueError(cat)
    classified.append(dict(domain='WasteSorting' if r['domain']=='waste' else 'TomatoHarvesting',
                           method='Ours' if r['method']=='pomdp' else 'KnowNo', run=r['run'],
                           classification=group, basis=basis, end_reason=r['end_reason'],
                           offered_candidates=' | '.join(offered), feasible_candidates=' | '.join(valid), context_evidence=evidence,
                           terminal_action=steps[-1][1] if steps else '',
                           source=str(path.relative_to(manuscript.parents[2])), raw_sha256=r['raw_sha256']))
assert len(classified) == 56
with (manuscript / 'asset/vlm_failure_classification.csv').open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(classified[0]))
    w.writeheader()
    w.writerows(classified)
for key, count in sorted(Counter((r['domain'], r['method'], r['classification']) for r in classified).items()):
    print(*key, count, sep=' | ')
