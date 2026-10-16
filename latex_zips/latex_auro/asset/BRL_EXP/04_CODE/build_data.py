"""Rebuild normalized runs and paper graph/table data from local raw logs.

Requires only Python's standard library. No dependency on experiments_final or
08_Delete_Me after organize_sources.py has populated the package.
"""
from pathlib import Path
from collections import defaultdict, Counter
from statistics import mean, stdev
import csv
import hashlib
import json
import math
import re
import summarize_human as human_parser

ROOT = Path(__file__).resolve().parents[1]
BASE, ABLATION, THRESHOLD = '01_BASELINE', '02_ABLATION', '03_THRESHOLD'
DOMAINS = ['WasteSorting', 'TomatoHarvesting']
DOMAIN = {'waste': DOMAINS[0], 'wastesorting': DOMAINS[0], 'tomato': DOMAINS[1]}
METHOD = {'knowno': 'KnowNo', 'introplan': 'IntroPlan', 'query_action_pomcp_selected': 'Query-Action',
          'pomdp': 'Ours', 'ours': 'Ours', 'random': 'Fully-Random',
          'ours_what_only': 'W1', 'cp_when': 'W2', 'value_when': 'W3',
          'ours_when_only': 'Q1', 'value_what': 'Q2'}
ORDER = ['KnowNo', 'IntroPlan', 'Query-Action', 'Fully-Random', 'W1', 'W2', 'W3', 'Q1', 'Q2', 'Ours']
METRICS = ['questions', 'query_probability', 'steps', 'plan_time_s', 'total_time_s',
           'interaction_time_s', 'execute_time_s', 'measured_search_time_s', 'planning_residual_time_s']
UNITS = {'questions': 'queries / run', 'query_probability': '%', 'steps': 'actions / run'}

def read(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def write(path, rows):
    assert rows, path
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

def checked_text(row):
    p = ROOT / row['raw_path']
    data = p.read_bytes()
    assert hashlib.sha256(data).hexdigest() == row['raw_sha256'], p
    return data.decode('utf-8-sig')

def question_counts(text):
    parts = human_parser.sections(text)
    counts = [(int(m[1]), len(re.findall(r'^- Q\d+:', m[2], re.M)))
              for m in re.finditer(r'^Step=(\d+):\s*\n(.*?)(?=^Step=|\Z)', parts.get('Questions', ''), re.M | re.S)]
    return sum(n for _, n in counts), len({i for i, n in counts if n})

def parse(row):
    text = checked_text(row)
    condition = row['condition']
    result = dict(environment=row['environment'], domain=DOMAIN[row['domain']], method=METHOD.get(condition, condition),
                  condition=condition, batch=row['batch'], scene=row.get('scene', ''),
                  participant='', run_id=row.get('run') or row['raw_path'],
                  status=row['status'], seed=row.get('seed', ''), threshold=row.get('threshold', ''),
                  raw_path=row['raw_path'], raw_sha256=row['raw_sha256'])
    for k in ['gamma', 'n_simulations', 'query_cost', 'model', 'qhat', 'generation_temperature']:
        result[k] = row.get(k, '')
    if row['status'] != 'ok':
        result.update(success=0, end_reason='execution_error')
        return result
    if condition in ['knowno', 'introplan'] and row['environment'] == 'Oracle':
        # Final JSON summary; generation/scoring JSON blocks remain untouched.
        marker = text.rfind('\nSummary:\n')
        assert marker >= 0, row['raw_path']
        data, _ = json.JSONDecoder().raw_decode(text[marker + len('\nSummary:\n'):].lstrip())
        steps, queries = int(data['planning_length']), int(data['question_count'])
        query_steps = len(set(re.findall(r'^Step (\d+) oracle answer:', text, re.M)))
        assert query_steps == queries
        search = sum(float(x) for x in re.findall(r'"elapsed_sec"\s*:\s*([0-9.eE+-]+)', text))
        result.update(success=int(data['success']), steps=steps, questions=queries,
                      query_steps=query_steps, query_probability=100 * query_steps / steps if steps else '',
                      total_time_s=float(data['total_elapsed_seconds']), plan_time_s=search,
                      measured_search_time_s=search, end_reason=data['stop_reason'])
    else:
        parts = human_parser.sections(text)
        summary, timing = parts['Plan Summary'], parts['Timing']
        steps = int(human_parser.value(summary, 'physical_steps', human_parser.value(summary, 'steps')))
        queries, query_steps = question_counts(text)
        assert queries == int(human_parser.value(summary, 'total_questions')), row['raw_path']
        search = human_parser.seconds(timing, 'search_time')
        total = human_parser.seconds(timing, 'total_time')
        interaction = human_parser.seconds(timing, 'interaction_time')
        execute = human_parser.seconds(timing, 'execute_time')
        result.update(success=int(human_parser.value(summary, 'success').lower() == 'true'),
                      steps=steps, questions=queries, query_steps=query_steps,
                      query_probability=100 * query_steps / steps if steps else '',
                      total_time_s=total, interaction_time_s=interaction, execute_time_s=execute,
                      plan_time_s=search, measured_search_time_s=search,
                      planning_residual_time_s=total-interaction-execute,
                      end_reason=human_parser.value(summary, 'end_reason'))
    # Check raw-derived quantities against the original run inventory.
    for a, b in [('questions', 'questions'), ('steps', 'steps'), ('success', 'success')]:
        ref = row.get(b, '')
        if ref != '':
            value = int(ref.lower() == 'true') if ref.lower() in ['true', 'false'] else float(ref)
            assert math.isclose(result[a], value, abs_tol=1e-7), (row['raw_path'], a, result[a], ref)
    return result

def human_runs():
    root = ROOT / BASE / '00_raw/human'
    results, original = [], []
    for participant in sorted(root.glob('p[0-9]*')):
        for path in sorted(participant.rglob('*.txt')):
            if path.parent.name not in {'planner_experiments', 'pomdp_experiments'}:
                continue
            parsed = human_parser.parse_result(path, participant.name)
            original.append({k: human_parser.display(parsed[k]) for k in human_parser.COLUMNS})
            parts = human_parser.sections(path.read_text(encoding='utf-8-sig'))
            execute = human_parser.seconds(parts['Timing'], 'execute_time')
            results.append(dict(environment='Human', domain=DOMAIN[parsed['domain']],
                                method=METHOD[parsed['condition']], condition=parsed['condition'],
                                batch='human_current', scene=parsed['scenario'], participant=participant.name,
                                run_id=str(path.parent.parent.relative_to(root)), status='ok',
                                success=int(parsed['success'] == 'TRUE'), questions=parsed['query'],
                                steps=parsed['steps'], query_probability=100 * parsed['query_rate'],
                                total_time_s=parsed['total_time'], interaction_time_s=parsed['interaction_time'],
                                execute_time_s=execute, plan_time_s=parsed['planning_time'],
                                measured_search_time_s=parsed['planning_time'],
                                planning_residual_time_s=parsed['total_time']-parsed['interaction_time']-execute,
                                raw_path=str(path.relative_to(ROOT)), raw_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    supplied = read(ROOT / BASE / '00_raw/source_inputs/human/human_summary.csv')
    def key(r):
        return tuple(r[k] for k in human_parser.COLUMNS)
    assert Counter(map(key, supplied)) == Counter(map(key, original)), 'Human raw logs differ from supplied summary'
    write(ROOT / BASE / '01_processed/human_summary.csv', original)
    return results

def wilson(n, k):
    z = 1.959963984540054
    p = k/n; den = 1+z*z/n
    center = (p+z*z/(2*n))/den
    half = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return max(0, 100*(center-half)), min(100, 100*(center+half))

def stats(runs, group_keys):
    groups = defaultdict(list)
    for r in runs:
        groups[tuple(r.get(k, '') for k in group_keys)].append(r)
    output = []
    for group, rows in groups.items():
        success = [r for r in rows if r['success']]
        n, k = len(rows), len(success)
        base = dict(zip(group_keys, group), total_n=n, success_n=k,
                    n_execution_errors=sum(r['status'] != 'ok' for r in rows), mode='success')
        low, high = wilson(n, k)
        output.append(dict(base, metric='success_rate', metric_n=n, mean=100*k/n,
                           sample_sd='', ci95_low=low, ci95_high=high, unit='%'))
        for metric in METRICS:
            vals = [float(r[metric]) for r in success if r.get(metric, '') != '']
            if not vals and not any(metric in r for r in rows):
                continue
            assert all(math.isfinite(v) for v in vals)
            output.append(dict(base, metric=metric, metric_n=len(vals), mean=mean(vals) if vals else '',
                               sample_sd=stdev(vals) if len(vals)>1 else '', ci95_low='', ci95_high='',
                               unit=UNITS.get(metric, 'seconds / run')))
    return sorted(output, key=lambda r: (DOMAINS.index(r['domain']), r.get('environment', ''),
                                         float(r.get('threshold') or 0), ORDER.index(r['method']) if r.get('method') in ORDER else 0,
                                         r['metric']))

def table_rows(statistics, group_keys, metrics):
    groups = defaultdict(dict)
    for r in statistics:
        groups[tuple(r[k] for k in group_keys)][r['metric']] = r
    output = []
    for key, lookup in groups.items():
        row = dict(zip(group_keys, key))
        s = lookup['success_rate']
        row.update(total_n=s['total_n'], success_n=s['success_n'], success_percent=s['mean'])
        for metric in metrics:
            if metric in lookup:
                r = lookup[metric]
                row.update({metric+'_mean': r['mean'], metric+'_sd': r['sample_sd'], metric+'_n': r['metric_n']})
        output.append(row)
    return output

def export_table(path, rows):
    write(path.with_suffix('.csv'), rows)
    # Standalone tabular; values stay numeric in CSV, formatted only for LaTeX.
    fields = [k for k in rows[0] if not k.endswith('_n') and not k.endswith('_sd')]
    def escape(v):
        return str(v).replace('_', r'\_').replace('%', r'\%')
    def cell(row, field):
        v = row.get(field, '')
        if v == '': return '--'
        if field.endswith('_mean'):
            sd = row.get(field.removesuffix('_mean')+'_sd', '')
            return f'${float(v):.2f}' + (f' \\pm {float(sd):.2f}$' if sd != '' else '$')
        if isinstance(v, float): return f'{v:.2f}'
        return escape(v)
    lines = [r'\begin{tabular}{' + 'l'*len(fields) + '}', r'\hline',
             ' & '.join(escape(f) for f in fields)+r' \\', r'\hline']
    lines += [' & '.join(cell(row, f) for f in fields)+r' \\' for row in rows]
    lines += [r'\hline', r'\end{tabular}']
    path.with_suffix('.tex').write_text('\n'.join(lines)+'\n')

def main():
    # Validate every raw copy, including retained source configurations.
    files = read(ROOT / 'source_files.csv')
    for r in files:
        assert hashlib.sha256((ROOT/r['destination']).read_bytes()).hexdigest() == r['sha256'], r['destination']
    index = read(ROOT / 'source_index.csv')
    parsed = [parse(r) for r in index]
    current_human = human_runs()
    shared = [r for r in parsed if r['batch']=='policy_when_what' and r['condition']=='ours']
    assert len(shared) == 400
    baseline = [r for r in parsed if (r['batch']=='oracle_baselines' and r['condition'] in
                {'knowno', 'introplan', 'query_action_pomcp_selected'}) or r['environment']=='VLM'] + shared + current_human
    ablation = [r for r in parsed if r['batch']=='policy_when_what' or
                (r['batch']=='random_when_what' and r['condition']!='ours')]
    threshold = [r for r in parsed if r['batch']=='threshold_sweep' and float(r['threshold']) != .8] + shared
    assert len(baseline)==1760+len(current_human) and len(ablation)==2800 and len(threshold)==4400
    validation = dict(source_files_verified=len(files), source_runs_verified=len(parsed),
                      human_participants=sorted({r['participant'] for r in current_human}),
                      shared_ours_runs=len(shared), current_human_runs=len(current_human), groups={})
    for section, rows, keys in [(BASE, baseline, ['environment', 'domain', 'method']),
                                (ABLATION, ablation, ['domain', 'method', 'condition']),
                                (THRESHOLD, threshold, ['domain', 'threshold'])]:
        write(ROOT/section/'01_processed/runs.csv', rows)
        statistics = stats(rows, keys)
        write(ROOT/section/'01_processed/statistics.csv', statistics)
        write(ROOT/section/'02_graph_data/statistics.csv', statistics)
        validation['groups'][section] = [r for r in statistics if r['metric']=='success_rate']
    base_stats = read(ROOT/BASE/'02_graph_data/statistics.csv')
    oracle = [r for r in base_stats if r['environment']=='Oracle']
    write(ROOT/BASE/'02_graph_data/exp_oracle_baselines_data.csv', oracle)
    write(ROOT/BASE/'02_graph_data/vlm_human_statistics.csv', [r for r in base_stats if r['environment']!='Oracle'])
    write(ROOT/BASE/'02_graph_data/response_comparison_statistics.csv', [r for r in base_stats if r['method'] in {'KnowNo', 'Ours'}])
    export_table(ROOT/BASE/'02_graph_data/oracle_table', table_rows(oracle, ['domain','method'],
                 ['questions','query_probability','steps','plan_time_s']))
    export_table(ROOT/BASE/'02_graph_data/physical_table', table_rows([r for r in base_stats if r['environment']!='Oracle'],
                 ['environment','domain','method'], ['questions','query_probability','steps','total_time_s','interaction_time_s','measured_search_time_s']))
    abl_stats = stats(ablation, ['domain','method','condition'])
    abl_table = table_rows(abl_stats, ['domain','method','condition'], ['questions','steps','plan_time_s'])
    export_table(ROOT/ABLATION/'02_graph_data/ablation_table', abl_table)
    # Match the original renderer's ablation input schema, with manuscript labels.
    write(ROOT/ABLATION/'02_graph_data/ablation_unified_by_domain.csv', [dict(domain='wastesorting' if r['domain']==DOMAINS[0] else 'tomato',
          condition=r['condition'], n_attempted=r['total_n'], successes=r['success_n'],
          success_percent_errors_as_failure=r['success_percent'],
          success_ci_low=wilson(r['total_n'],r['success_n'])[0], success_ci_high=wilson(r['total_n'],r['success_n'])[1],
          questions_success_only_mean=r['questions_mean'], questions_success_only_sd=r['questions_sd']) for r in abl_table])
    th_table = table_rows(stats(threshold, ['domain','threshold']), ['domain','threshold'], ['questions'])
    export_table(ROOT/THRESHOLD/'02_graph_data/threshold_table', th_table)
    write(ROOT/THRESHOLD/'02_graph_data/manuscript_threshold_by_domain.csv', [dict(domain=r['domain'],threshold=r['threshold'],
          n=r['total_n'], successes=r['success_n'], success_percent=r['success_percent'],
          questions_mean=r['questions_mean'], questions_sd=r['questions_sd']) for r in th_table])
    # Successful human metrics, one trial = one observation; pooled ratio retained separately in aggregate.
    src = ROOT/BASE/'01_processed/human_summary.csv'
    import subprocess, sys
    subprocess.run([sys.executable, str(Path(__file__).with_name('aggregate_human_summary.py')),
                    '--input', str(src), '--output', str(src.with_name('human_aggregate.csv'))], check=True, stdout=subprocess.DEVNULL)
    snapshot = ROOT/BASE/'00_raw/source_inputs/manuscript_snapshot/exp_oracle_baselines_data.csv'
    comparisons = []
    look = {(r['domain'],r['method'],r['metric']):r for r in oracle}
    for old in read(snapshot):
        new = look[old['domain'],old['method'],old['metric']]
        for key in ['mean','sample_sd','ci95_low','ci95_high']:
            if old[key] and new[key]:
                assert math.isclose(float(old[key]),float(new[key]),abs_tol=1e-8), (old,key,new)
        comparisons.append(dict(domain=old['domain'],method=old['method'],metric=old['metric'],matches=True))
    validation['oracle_paper_metrics_matched'] = len(comparisons)
    # Compare all original ablation/threshold plotted values to rebuilt data.
    for filename, current, keys, values in [
        ('ablation_unified_by_domain.csv', read(ROOT/ABLATION/'02_graph_data/ablation_unified_by_domain.csv'),
         ['domain','condition'], ['success_percent_errors_as_failure','questions_success_only_mean','questions_success_only_sd']),
        ('manuscript_threshold_by_domain.csv', read(ROOT/THRESHOLD/'02_graph_data/manuscript_threshold_by_domain.csv'),
         ['domain','threshold'], ['success_percent','questions_mean','questions_sd'])]:
        original = {tuple(r[k] for k in keys):r for r in read(snapshot.parent/filename)}
        for row in current:
            old = original[tuple(row[k] for k in keys)]
            for key in values:
                assert math.isclose(float(row[key]),float(old[key]),abs_tol=1e-8), (filename,row,key,old[key])
    validation['ablation_threshold_paper_metrics_matched'] = True
    previous_vlm = {(r['domain'],r['method'],r['metric']):r for r in read(snapshot.parent/'vlm_human_statistics.csv')
                    if r['environment']=='VLM' and r['mode']=='success'}
    for row in base_stats:
        if row['environment']=='VLM' and row['metric'] in {'success_rate','questions','query_probability','steps','total_time_s','interaction_time_s','planning_residual_time_s'}:
            old = previous_vlm[row['domain'],row['method'],row['metric']]
            for key in ['mean','sample_sd']:
                if row[key] and old[key]:
                    assert math.isclose(float(row[key]),float(old[key]),abs_tol=1e-8), (row,key,old[key])
    validation['vlm_source_statistics_matched'] = True
    (ROOT/'validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in validation.items() if k!='groups'},ensure_ascii=False,indent=2))

if __name__ == '__main__':
    main()
