"""Build separate Oracle metric PDFs from the manuscript reference and raw logs."""
from pathlib import Path
import csv
import hashlib
import math
import os
import re
from statistics import mean, stdev

os.environ.setdefault('MPLCONFIGDIR', '/tmp/brl-paper-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MANUSCRIPT = Path(__file__).resolve().parents[1]
EXPERIMENTS = MANUSCRIPT.parents[1] / 'experiments_final'
SUMMARY = EXPERIMENTS / '04_Total_Summary'
BASELINE = EXPERIMENTS / '01_Oracle/03_baseline'
DOMAINS = ['WasteSorting', 'TomatoHarvesting']
METHODS = ['KnowNo', 'IntroPlan', 'Query-Action', 'Ours']

def read(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

rows = [r for r in read(SUMMARY / 'data/manuscript_comparison_statistics.csv')
        if r['scope'] == 'oracle_all_scenes' and r['mode'] == 'success']
lookup = {(r['domain'], r['method'], r['metric']): r for r in rows}
probabilities = {}
plan_times = {}
for r in read(BASELINE / '01_processed/episodes.csv'):
    if r['condition'] not in ['knowno', 'introplan', 'query_action_pomcp_selected']:
        continue
    path = BASELINE / '00_raw' / r['raw_source'].split('/00_raw/', 1)[1]
    data = path.read_bytes()
    assert hashlib.sha256(data).hexdigest() == r['raw_sha256'], path
    text = data.decode()
    if r['condition'] in ['knowno', 'introplan']:
        ids = list(map(int, re.findall(r'^Step (\d+) oracle answer:', text, re.M)))
        count = len(ids)
        method = 'IntroPlan' if r['condition'] == 'introplan' else 'KnowNo'
        elapsed = re.findall(r'"elapsed_sec"\s*:\s*([0-9.eE+-]+)', text)
        assert elapsed, path
        plan_time = sum(map(float, elapsed))
    else:
        section = text.split('[Questions]')[1].split('[Final Knowledge]')[0]
        blocks = list(re.finditer(r'^Step=(\d+):\n(.*?)(?=^Step=|\Z)', section, re.M | re.S))
        counts = [(int(m[1]), len(re.findall(r'^- Q\d+:', m[2], re.M))) for m in blocks]
        ids = [i for i, n in counts if n]
        count = sum(n for _, n in counts)
        method = 'Query-Action'
        plan_time = float(re.search(r'^search_time: total=([0-9.]+)s', text, re.M)[1])
    assert count == int(r['questions']), path
    if r['success'] == '1':
        value = 100 * len(set(ids)) / float(r['steps'])
        assert 0 <= value <= 100, path
        domain = 'WasteSorting' if r['domain'] == 'wastesorting' else 'TomatoHarvesting'
        probabilities.setdefault((domain, method), []).append(value)
        plan_times.setdefault((domain, method), []).append(plan_time)
for (domain, method), values in probabilities.items():
    reference = lookup[domain, method, 'questions']
    assert len(values) == int(reference['success_n'])
    lookup[domain, method, 'query_probability'] = {
        **reference, 'metric': 'query_probability', 'mean': mean(values), 'sample_sd': stdev(values)}

for r in read(SUMMARY / 'data/manuscript_oracle_reference_runs.csv'):
    if r['success'] != '1':
        continue
    path = EXPERIMENTS / r['source']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == r['raw_sha256'], path
    text = path.read_text()
    value = float(re.search(r'^search_time: total=([0-9.]+)s', text, re.M)[1])
    plan_times.setdefault((r['domain'], 'Ours'), []).append(value)
for (domain, method), values in plan_times.items():
    reference = lookup[domain, method, 'questions']
    assert len(values) == int(reference['success_n'])
    lookup[domain, method, 'plan_time_s'] = {
        **reference, 'metric': 'plan_time_s', 'mean': mean(values), 'sample_sd': stdev(values)}

metrics = [('success_rate', 'Success rate', '%'),
           ('questions', 'Query count', 'Queries per trial'),
           ('query_probability', 'Query probability', '% of physical steps'),
           ('plan_time_s', 'Plan time', 'Seconds per trial')]
output = []
for domain in DOMAINS:
    for method in METHODS:
        for metric, _, _ in metrics + [('steps', 'Plan length', '')]:
            r = lookup[domain, method, metric]
            output.append({k: r.get(k, '') for k in ['domain', 'method', 'metric', 'total_n', 'success_n', 'mean', 'sample_sd', 'ci95_low', 'ci95_high']})
with (MANUSCRIPT / 'figures/exp_oracle_baselines_data.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(output[0]))
    writer.writeheader()
    writer.writerows(output)

# One graph per PDF. LaTeX composes the separate panels with subfigure.
from plot_experiment_panels import render_oracle
render_oracle(output)
for domain in DOMAINS:
    for method in METHODS:
        r = lookup[domain, method, 'plan_time_s']
        print(f"{domain} {method} plan time {float(r['mean']):.2f} ± {float(r['sample_sd']):.2f}")
