"""Build separate step-20 manuscript data, figures, and experimental sections.

Original experimental TeX, plots, and raw data remain unchanged. Success at
physical action 20 / POMCP-Q query 30 is retained. Oracle trajectories alone
are re-scored; all reported costs use retained successful trials.
"""
from pathlib import Path
from statistics import mean
import hashlib
import json
import math
import re
import shutil
import sys

PAPER = Path(__file__).resolve().parents[1]
PACKAGE = PAPER / 'asset/BRL_EXP'
STEP = PACKAGE / '05_STEP20'
CODE = PACKAGE / '04_CODE'
sys.path.insert(0, str(CODE))
import build_data as build
import plot_experiment_panels as plot


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate_data():
    all_statistics, source_files, checked = {}, {}, {}
    for section, keys in [
        ('01_BASELINE', ['environment', 'domain', 'method']),
        ('02_ABLATION', ['domain', 'method', 'condition']),
        ('03_THRESHOLD', ['domain', 'threshold']),
    ]:
        source = PACKAGE / section / '01_processed/runs.csv'
        source_files[str(source.relative_to(PAPER))] = digest(source)
        rows = []
        for source_row in build.read(source):
            if source_row['environment'] != 'Oracle':
                continue
            raw = PACKAGE / source_row['raw_path']
            if str(raw) not in checked:
                checked[str(raw)] = digest(raw)
            require(checked[str(raw)] == source_row['raw_sha256'], f'Raw changed: {raw}')
            row = dict(source_row)
            old_success = int(row['success'])
            new_success = int(old_success and float(row['steps']) <= 20 and
                              (row['condition'] != 'query_action_pomcp_selected' or float(row['questions']) <= 30))
            row.update(original_success=old_success, success=new_success,
                       max_physical_steps=20,
                       max_queries=30 if row['condition'] == 'query_action_pomcp_selected' else '',
                       original_end_reason=row.get('end_reason', ''))
            if old_success and not new_success:
                row['end_reason'] = 'MAX QUERY' if float(row['questions']) > 30 and row['condition'] == 'query_action_pomcp_selected' else 'MAX STEP'
            rows.append(row)
        require(len(rows) == {'01_BASELINE': 1600, '02_ABLATION': 2800, '03_THRESHOLD': 4400}[section], f'Wrong sample size: {section}')
        build.write(STEP / section / '01_processed/runs.csv', rows)
        statistics = build.stats(rows, keys)
        build.write(STEP / section / '01_processed/statistics.csv', statistics)
        build.write(STEP / section / '02_graph_data/statistics.csv', statistics)
        all_statistics[section] = statistics

    # Verify the exact baseline and ablation batches already agreed with the user.
    canonical = PAPER.parents[1] / 'experiments_final/01_Oracle'
    if canonical.exists():
        baseline = build.read(canonical / '03_baseline/01_processed/episodes.csv')
        expected = {(r['condition'], r['raw_sha256']): int(r['success']) for r in baseline}
        for r in build.read(STEP / '01_BASELINE/01_processed/runs.csv'):
            require(int(r['success']) == expected[r['condition'], r['raw_sha256']], 'Baseline differs from canonical source')
        ablation = build.read(canonical / '04_ablation_step20/episodes.csv')
        expected = {(r['source_condition'], r['raw_sha256']): int(r['success']) for r in ablation}
        for r in build.read(STEP / '02_ABLATION/01_processed/runs.csv'):
            require(int(r['success']) == expected[r['condition'], r['raw_sha256']], 'Ablation differs from canonical source')

    base = all_statistics['01_BASELINE']
    build.write(STEP / '01_BASELINE/02_graph_data/exp_oracle_baselines_data.csv', base)
    ablation = build.table_rows(all_statistics['02_ABLATION'], ['domain', 'method', 'condition'], ['questions', 'steps', 'plan_time_s'])
    build.export_table(STEP / '02_ABLATION/02_graph_data/ablation_table', ablation)
    build.write(STEP / '02_ABLATION/02_graph_data/ablation_unified_by_domain.csv', [
        dict(domain='wastesorting' if r['domain'] == 'WasteSorting' else 'tomato', condition=r['condition'],
             n_attempted=r['total_n'], successes=r['success_n'], success_percent_errors_as_failure=r['success_percent'],
             success_ci_low=build.wilson(r['total_n'], r['success_n'])[0], success_ci_high=build.wilson(r['total_n'], r['success_n'])[1],
             questions_success_only_mean=r['questions_mean'], questions_success_only_sd=r['questions_sd']) for r in ablation])
    threshold = build.table_rows(all_statistics['03_THRESHOLD'], ['domain', 'threshold'], ['questions'])
    build.export_table(STEP / '03_THRESHOLD/02_graph_data/threshold_table', threshold)
    build.write(STEP / '03_THRESHOLD/02_graph_data/manuscript_threshold_by_domain.csv', [
        dict(domain=r['domain'], threshold=r['threshold'], n=r['total_n'], successes=r['success_n'],
             success_percent=r['success_percent'], questions_mean=r['questions_mean'], questions_sd=r['questions_sd']) for r in threshold])

    # Reuse the existing plotting style, exporting into distinct directories.
    plot.ROOT = STEP
    plot.BASE = STEP / '01_BASELINE/02_graph_data'
    plot.ABLATION = STEP / '02_ABLATION/02_graph_data'
    plot.THRESHOLD = STEP / '03_THRESHOLD/02_graph_data'
    plot.MANIFEST.clear()
    plot.style()
    plot.render_oracle(base)
    plot.render_ablation()
    plot.render_threshold()
    groups = {'01_BASELINE': '04_ORACLE_BASELINE_STEP20', '02_ABLATION': '08_ABLATION_STEP20', '03_THRESHOLD': '09_THRESHOLD_STEP20'}
    for item in plot.MANIFEST:
        source = STEP / item['file']
        target = PAPER / 'figures' / groups[Path(item['file']).parts[0]] / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        item['published_file'] = str(target.relative_to(PAPER))
        item['sha256'] = digest(target)
    (STEP / 'figure_manifest.json').write_text(json.dumps(plot.MANIFEST, indent=2) + '\n')

    ours_hashes = {}
    for section in ['01_BASELINE', '02_ABLATION']:
        rr = [r for r in build.read(STEP / section / '01_processed/runs.csv') if r.get('method') == 'Ours' and
              (section != '03_THRESHOLD' or float(r['threshold']) == .8)]
        ours_hashes[section] = {(r['domain'], r['seed'], r['raw_sha256'], r['success']) for r in rr}
    require(ours_hashes['02_ABLATION'] == ours_hashes['01_BASELINE'], 'Shared Ours reference differs between baseline and ablation')
    threshold_rows = build.read(STEP / '03_THRESHOLD/01_processed/runs.csv')
    require(all(r['batch'] == 'threshold_sweep' for r in threshold_rows), 'Threshold contains runs outside the original sweep')
    require(not ({r['seed'] for r in threshold_rows} & {r['seed'] for r in build.read(STEP / '01_BASELINE/01_processed/runs.csv') if r['method'] == 'Ours'}), 'Threshold sweep and Ours evaluation seeds overlap')
    validation = dict(boundary_rule='physical actions <=20; POMCP-Q queries <=30; boundary success allowed',
                      analysis='retrospective rescoring; successful-run costs recomputed',
                      original_source_files=source_files, unique_raw_files_verified=len(checked),
                      shared_ours_identical=True, shared_ours_sections=['01_BASELINE', '02_ABLATION'],
                      threshold_original_sweep=True, threshold_seeds_disjoint_from_ours=True,
                      figures=len(plot.MANIFEST),
                      groups={s: [r for r in rows if r['metric'] == 'success_rate'] for s, rows in all_statistics.items()})
    (STEP / 'validation.json').write_text(json.dumps(validation, indent=2) + '\n')
    return all_statistics


def lookup(statistics, section, domain, method, metric):
    return next(r for r in statistics[section] if r['domain'] == domain and r.get('method') == method and r['metric'] == metric)


def threshold_lookup(statistics, domain, threshold, metric):
    return next(r for r in statistics['03_THRESHOLD'] if r['domain'] == domain and float(r['threshold']) == threshold and r['metric'] == metric)


def metric_mean(statistics, section, domain, method, metric):
    return float(lookup(statistics, section, domain, method, metric)['mean'])


def substitute_numbers(text, values):
    pattern = r'(?<![\d.])(?:' + '|'.join(re.escape(k) for k in sorted(values, key=len, reverse=True)) + r')(?![\d.])'
    return re.sub(pattern, lambda m: values[m[0]], text)


def numeric_cell(cell, values):
    require(len(re.findall(r'\d+\.\d+', cell)) == len(values), f'Unexpected numeric cell: {cell}')
    iterator = iter(values)
    return re.sub(r'\d+\.\d+', lambda _: f'{next(iterator):.2f}', cell)


def update_tex(statistics):
    originals = [PAPER / 'section/05_exp_setting.tex', PAPER / 'section/05_exp_system.tex',
                 PAPER / 'appendix/02_detailedEXP.tex']
    hashes = {str(p.relative_to(PAPER)): digest(p) for p in originals}
    source = originals[1].read_text()
    lines = source.splitlines()
    updated = []
    oracle_domain = None
    in_ablation = False
    in_threshold = False
    method_names = {'KnowNo': 'KnowNo', 'IntroPlan': 'IntroPlan', 'POMCP-Q': 'Query-Action', 'Ours': 'Ours'}
    ablation_methods = {'Fully-Random': 'Fully-Random', 'T1': 'W1', 'T2': 'W2', 'T3': 'W3', 'S1': 'Q1', 'S2': 'Q2', 'Ours': 'Ours'}
    for line in lines:
        if line.startswith(r'\subsubsection{Ablation Study}'):
            in_ablation = True
        if line.startswith(r'\subsubsection{Threshold Experiments}'):
            in_ablation, in_threshold = False, True
        if line.startswith(r'\subsubsection{Interviews}'):
            in_threshold = False
        if r'{Oracle\\Sort}' in line:
            oracle_domain = 'WasteSorting'
        elif r'{Oracle\\Harvest}' in line:
            oracle_domain = 'TomatoHarvesting'
        elif r'{VLM\\Sort}' in line:
            oracle_domain = None
        if oracle_domain and ' & ' in line and not line.startswith('%'):
            cells = line.split(' & ')
            if len(cells) == 9:
                method = next((v for k, v in method_names.items() if k in cells[1]), None)
                if method:
                    success = metric_mean(statistics, '01_BASELINE', oracle_domain, method, 'success_rate')
                    cells[2] = numeric_cell(cells[2], [success])
                    for index, metric in [(3, 'questions'), (4, 'query_probability'), (5, 'steps'), (6, 'plan_time_s'), (8, 'plan_time_s')]:
                        record = lookup(statistics, '01_BASELINE', oracle_domain, method, metric)
                        cells[index] = numeric_cell(cells[index], [float(record['mean']), float(record['sample_sd'])])
                    line = ' & '.join(cells)
        if in_ablation and ' & ' in line and not line.startswith('%'):
            cells = line.split(' & ')
            label = cells[0].replace(r'\textbf{Ours}', 'Ours')
            if label in ablation_methods and len(cells) == 7:
                method = ablation_methods[label]
                for domain, success_column, queries_column in [('WasteSorting', 3, 4), ('TomatoHarvesting', 5, 6)]:
                    success = metric_mean(statistics, '02_ABLATION', domain, method, 'success_rate')
                    queries = lookup(statistics, '02_ABLATION', domain, method, 'questions')
                    cells[success_column] = numeric_cell(cells[success_column], [success])
                    cells[queries_column] = numeric_cell(cells[queries_column], [float(queries['mean']), float(queries['sample_sd'])])
                line = ' & '.join(cells)
        if line.startswith('As shown in Table~'):
            line = substitute_numbers(line, {'98.00': '97.50', '87.00': '76.00', '57.50': '56.00'})
        elif line.startswith('Ours uses more queries than KnowNo'):
            line = substitute_numbers(line, {'98.00': '97.50'})
        elif line.startswith('The timing comparison evaluates'):
            head, tail = line.split('In Harvest,', 1)
            tail = substitute_numbers(tail, {'73.50': '72.00', '63.00': '60.50', '68.50': '66.50', '98.00': '97.50',
                                            '9.04': '9.08', '7.43': '7.44', '4.05': '3.91', '12.16': '12.02'})
            line = head + 'In Harvest,' + tail
        elif line.startswith('These results show that task success depends'):
            ours = metric_mean(statistics, '02_ABLATION', 'TomatoHarvesting', 'Ours', 'questions')
            reward = metric_mean(statistics, '02_ABLATION', 'TomatoHarvesting', 'W3', 'questions')
            line = substitute_numbers(line, {'32.71': f'{100*(reward/ours-1):.2f}', '29.50': '31.00'})
        elif line.startswith('With confidence-based query timing fixed, different query selection strategies'):
            line = line.replace('Fully-Random achieves $73.50\\%$ success in both domains',
                                'Fully-Random achieves $73.50\\%$ success in Sort and $72.00\\%$ in Harvest')
            line = line.replace('In Harvest, S1 and S2 achieve $97.50\\%$ success',
                                'In Harvest, S1 and S2 achieve $96.50\\%$ and $97.00\\%$ success, respectively,')
            line = substitute_numbers(line, {'97.50': '96.50', '98.00': '97.50', '9.04': '9.08', '12.68': '12.70', '15.25': '15.27'})
        elif line.startswith('Confidence timing maintains'):
            values = {}
            for domain, old_random, old_reward in [('WasteSorting', '48.54', '65.64'), ('TomatoHarvesting', '27.72', '39.92')]:
                ours = metric_mean(statistics, '02_ABLATION', domain, 'Ours', 'questions')
                q1 = metric_mean(statistics, '02_ABLATION', domain, 'Q1', 'questions')
                q2 = metric_mean(statistics, '02_ABLATION', domain, 'Q2', 'questions')
                values.update({old_random: f'{100*(1-ours/q1):.2f}', old_reward: f'{100*(1-ours/q2):.2f}'})
            values['97.50'] = '96.50'
            line = substitute_numbers(line, values)
        elif in_threshold and re.match(r'^% (?:0\.\d|1\.0) & ', line):
            cells = line.split(' & ')
            th = float(cells[0].removeprefix('% '))
            for domain, success_column, queries_column in [('WasteSorting', 1, 2), ('TomatoHarvesting', 3, 4)]:
                rate = threshold_lookup(statistics, domain, th, 'success_rate')
                queries = threshold_lookup(statistics, domain, th, 'questions')
                cells[success_column] = numeric_cell(cells[success_column], [float(rate['mean'])])
                cells[queries_column] = numeric_cell(cells[queries_column], [float(queries['mean']), float(queries['sample_sd'])])
            line = ' & '.join(cells)
        for old, new in [('04_ORACLE_BASELINE', '04_ORACLE_BASELINE_STEP20'), ('08_ABLATION', '08_ABLATION_STEP20'), ('09_THRESHOLD', '09_THRESHOLD_STEP20')]:
            line = line.replace(f'figures/{old}/', f'figures/{new}/')
        updated.append(line)
    system = '\n'.join(updated) + '\n'
    system = system.replace('% Numerical reference is the manuscript summary in experiments_final/04_Total_Summary.',
                            '% Numerical reference: asset/BRL_EXP/05_STEP20; originals retained in 05_exp_system.tex.')
    (PAPER / 'section/05_exp_system_step20.tex').write_text(system)

    setting = originals[0].read_text()
    needle = r'Appendix~\ref{appx:settings} provides the baseline configurations and additional experimental details.'
    require(setting.count(needle) == 1, 'Implementation paragraph changed')
    setting = setting.replace(needle, r'Oracle trajectories are evaluated with limits of $20$ physical actions and, for POMCP-Q, $30$ queries. ' + needle)
    (PAPER / 'section/05_exp_setting_step20.tex').write_text(setting)

    appendix = originals[2].read_text()
    old_limit = r'Oracle trials allow $50$ physical actions, except that POMCP-Q counts both queries and physical actions toward its $50$-decision limit.'
    new_limit = r'Oracle trajectories are re-evaluated with a limit of $20$ physical actions. POMCP-Q has a separate limit of $30$ queries. A trial fails if either budget would be exceeded; success on the $20$th physical action or $30$th query is retained.'
    require(appendix.count(old_limit) == 1, 'Appendix budget paragraph changed')
    appendix = appendix.replace(old_limit, new_limit)
    (PAPER / 'appendix/02_detailedEXP_step20.tex').write_text(appendix)

    main_path = PAPER / 'main.tex'
    main = main_path.read_text()
    for old, new in [('section/05_exp_setting', 'section/05_exp_setting_step20'),
                     ('section/05_exp_system', 'section/05_exp_system_step20'),
                     ('appendix/02_detailedEXP.tex', 'appendix/02_detailedEXP_step20.tex')]:
        # Also permit repeated regeneration after main already uses the new files.
        main = re.sub(r'(\\input\{)' + re.escape(old) + r'(\})', lambda m: m[1] + new + m[2], main)
    main_lines = []
    for line in main.splitlines():
        if line.startswith(r'\abstract{'):
            line = substitute_numbers(line, {'87.00': '76.00', '98.00': '97.50', '57.50': '56.00'})
        main_lines.append(line)
    main_path.write_text('\n'.join(main_lines) + '\n')
    for p in originals:
        require(digest(p) == hashes[str(p.relative_to(PAPER))], f'Original TeX modified: {p}')
    (STEP / 'original_tex_hashes.json').write_text(json.dumps(hashes, indent=2) + '\n')
    (STEP / 'README.txt').write_text(
        '20회 제한 수정본 (2026-10-23)\n'
        '물리 행동 20회까지, POMCP-Q 질문 30회까지 허용한다. 경계의 성공을 유지한다.\n'
        'Oracle baseline / ablation / threshold의 기존 실행 경로를 재판정했다.\n'
        '성공 시행의 질의 수·질의확률·계획 길이·기록된 계획 시간을 새 성공 집합에서 재계산했다.\n'
        'VLM·Human 데이터와 원본 tex/그림/원자료는 수정하지 않는다.\n'
        '수정 tex: section/05_exp_setting_step20.tex, section/05_exp_system_step20.tex\n'
        '수정 부록: appendix/02_detailedEXP_step20.tex\n'
        '수정 그림: figures/04_ORACLE_BASELINE_STEP20, 08_ABLATION_STEP20, 09_THRESHOLD_STEP20\n'
        'main.tex가 수정본을 참조한다. 원본 main.tex/PDF와 실험 tex·그림 백업은\n'
        '02_BRL_CODE/experiments_backups/20261023_manuscript_step20/에 있다.\n'
        '재생성: python3 scripts/update_step20_results.py\n')
    print('Updated experimental TeX copies and main entry; original experimental TeX preserved.')


if __name__ == '__main__':
    statistics = generate_data()
    update_tex(statistics)
