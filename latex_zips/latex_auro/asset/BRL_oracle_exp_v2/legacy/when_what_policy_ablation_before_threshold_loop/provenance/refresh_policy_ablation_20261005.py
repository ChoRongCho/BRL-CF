"""Freeze, analyze and publish the 2026-10-05 Value-policy rerun.

Run from the project root with Python 3.10+ and matplotlib.
Preserves the previous published package under BRL_oracle_exp_v2/legacy.
"""
from pathlib import Path
from collections import Counter
import csv
import hashlib
import json
import math
import shutil
import statistics

import pipeline

ROOT = pipeline.PROJECT
BUNDLE = ROOT / 'latex_zips/latex_auro/asset/BRL_oracle_exp_v2'
DEST = BUNDLE / '02_when_what_policy_ablation'
ARCHIVE = BUNDLE / 'legacy/when_what_policy_ablation_before_20261005'
RUN = ROOT / 'experiments_logs/when_what_policy_ablation/20261005_190346_519071'
PACKAGE = pipeline.BASE / 'when_what_policy_ablation_20261005'


def read(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact_p(wins, losses):
    n = wins + losses
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(min(wins, losses)+1)) / 2**n) if n else 1.0


def main():
    previous = ARCHIVE if ARCHIVE.exists() else DEST
    old_manifest = json.loads((previous / 'manifest.json').read_text())
    runs = read(RUN / 'runs.csv')
    assert len(runs) == 800 and all(r['status'] == 'complete' for r in runs)
    assert Counter((r['domain'], r['condition'], int(r['scene'])) for r in runs) == Counter({
        (d, c, s): 40 for d in ('tomato', 'wastesorting')
        for c in ('value_when', 'value_what') for s in range(1, 6)})
    PACKAGE.mkdir(exist_ok=True)
    sources = []
    for item in old_manifest['sources']:
        if item['condition'] not in ('ours', 'cp_when'):
            continue
        relative = Path(item['source_file']).relative_to(
            Path('experiments_logs/analysis_recent/when_what_policy_ablation_20260921/00_raw'))
        original = previous / '00_raw' / relative
        if not original.is_file():
            original = ROOT / item['source_file']
        assert digest(original) == item['sha256']
        target = PACKAGE / '00_raw' / relative
        shutil.copytree(original.parent, target.parent, dirs_exist_ok=True)
        sources.append(dict(item, source_file=str(target.relative_to(ROOT))))
    traces = {}
    diagnostics = []
    for r in runs:
        folder = Path(r['log_dir'])
        logs = list(folder.glob('exp_*.txt'))
        assert len(logs) == 1
        data = json.loads((folder / 'policy_trace.json').read_text())
        meta = data['meta']
        assert meta['domain'] == r['domain'] and meta['condition'] == r['condition']
        assert meta['seed'] == int(r['seed'])
        assert meta['gamma'] == (0.5 if r['domain'] == 'tomato' else 0.9)
        assert meta['query_cost'] == 0.0 and meta['n_simulations'] == 100
        target_folder = PACKAGE / '00_raw' / folder.relative_to(RUN)
        shutil.copytree(folder, target_folder, dirs_exist_ok=True)
        target = target_folder / logs[0].name
        sources.append(dict(source_file=str(target.relative_to(ROOT)), sha256=digest(target),
            domain=r['domain'], condition=r['condition'], scene=f"{int(r['scene']):02}",
            iteration=r['iteration'], seed=r['seed'], exit_code='0'))
        traces[(r['domain'], r['condition'], int(r['scene']), r['seed'])] = data
        for step_index, step in enumerate(data['policy_traces'], 1):
            questions = step['questions']
            if r['condition'] == 'value_when':
                assert len(questions) <= 1
            diagnostics.append(dict(domain=r['domain'], condition=r['condition'], scene=r['scene'],
                seed=r['seed'], step=step_index, questions=len(questions),
                stop_reason=step['stop_reason'], final_confidence=step['final_confidence']))

    notes = ('Ours・Action-CP When은 2026-09-21~22의 각 400회를 유지하고, Value-When・Value-What은 '
        '2026-10-05 배치의 각 400회로 교체했다. Value 조건은 Tomato gamma=0.5, Waste gamma=0.9, '
        'query_cost=0.0, n_simulations=100이다. Ours・CP-When은 gamma=0.2이다. '
        'gamma는 질문 가치 평가기와 실제 행동용 POMCP 모두에 적용됐다. Value-When의 물리 행동당 '
        '최대 1개 질문 제한은 유지됐다. 따라서 질문 시점만 바꾼 통제 비교로 해석하지 않는다. '
        'CP-When의 기존 형식 파싱 오류 4건은 성공률에서 실패로 집계한다. '
        '이전 Value 배치와 새 배치는 합산하지 않고 별도 paired 비교로 제공한다.')
    config = dict(old_manifest, title='When–What policy comparison (Value rerun: 2026-10-05)',
        sources=sources, notes=notes, status='analyzed_with_design_limitations',
        status_note='새 Value 실행 800/800 정상 완료; 기존 Ours・CP 800개 참조. 질문 구조 및 gamma 차이를 명시한 비교.',
        key_findings=[], rerun_root=str(RUN.relative_to(ROOT)),
        reference_batch='when_what_policy_ablation_20260921',
        previous_published_package=str(ARCHIVE.relative_to(ROOT)))
    (PACKAGE / 'manifest.json').write_text(json.dumps(config, ensure_ascii=False, indent=2)+'\n')
    prov = PACKAGE / 'provenance'
    prov.mkdir(exist_ok=True)
    shutil.copy2(RUN / 'manifest.json', prov / 'run_manifest.json')
    shutil.copy2(RUN / 'runs.csv', prov / 'runs.csv')
    for name in ('run_when_what_policy_ablation.sh', 'iterate_when_what_policy_ablation.sh'):
        shutil.copy2(ROOT / 'run' / name, prov / name)
    for name in ('runner.py', 'policies.py', 'query_episode.py', 'value_evaluator.py'):
        shutil.copy2(ROOT / 'scripts/ablation/when_what_policy_ablation/script' / name, prov / name)
    (prov / 'README.md').write_text('Shell/Python snapshots were captured at analysis time, not automatically at experiment launch. Raw policy traces are the evidence for executed parameters and question limits.\n')
    shutil.copy2(Path(__file__), prov / Path(__file__).name)

    pipeline.stage1(PACKAGE)
    episodes = read(PACKAGE / '01_processed/episodes.csv')
    for r in episodes:
        if r['condition'].startswith('value_'):
            data = traces[(r['domain'], r['condition'], int(r['scene']), r['seed'])]
            assert r['status'] == 'ok'
            assert int(r['success']) == int(data['success'])
            assert int(r['questions']) == data['total_questions']
            assert int(r['steps']) == len(data['policy_traces'])
    pipeline.stage2(PACKAGE)
    summary = read(PACKAGE / '01_processed/summary.csv')
    lookup = {(r['domain'], r['condition']): r for r in summary}
    vw, vt = lookup['all', 'value_when'], lookup['all', 'value_what']
    config['key_findings'] = [
        f"Value-When 성공률 {float(vw['success_percent']):.2f}% ({vw['successes']}/400), 전체 평균 질문 {float(vw['questions_mean']):.3f}회, 성공 실행 평균 {float(vw['questions_success_only_mean']):.3f}회.",
        f"Value-What 성공률 {float(vt['success_percent']):.2f}% ({vt['successes']}/400), 전체 평균 질문 {float(vt['questions_mean']):.3f}회, 성공 실행 평균 {float(vt['questions_success_only_mean']):.3f}회.",
        '설정 변경 전 대비 Value-When 성공률은 60.25%에서 70.00%로, Value-What은 98.25%에서 98.75%로 변했다. gamma와 cost를 동시에 바꿨으므로 개별 효과를 분리하지 않는다.',
        'Value-When의 질문 1개 제한은 실제 모든 step에서 확인됐다. 의도했던 공통 threshold 반복 구조를 복구한 실험은 아니다.']
    (PACKAGE / 'manifest.json').write_text(json.dumps(config, ensure_ascii=False, indent=2)+'\n')
    pipeline.stage2(PACKAGE)
    pipeline.stage3(PACKAGE)
    pipeline.docs(PACKAGE)
    readme = PACKAGE / 'README.md'
    text = readme.read_text().replace(
        f'python3 experiments_logs/analysis_recent/scripts/pipeline.py --package {PACKAGE.name} --stage all',
        '/home/fr/miniconda3/envs/brl/bin/python experiments_logs/analysis_recent/scripts/refresh_policy_ablation_20261005.py')
    text = text.replace('단계별로 `--stage 1`, `--stage 2`, `--stage 3` 실행 가능. 원본 해시가 바뀌면 자동 중단. 오래된 배치와 최신 배치는 합치지 않음.',
        '이 명령은 원본 해시・800회 설정・trace와 집계 일치를 검증하고 분석과 LaTeX 패키지를 다시 생성한다. 이전 Value와 새 Value를 합산하지 않는다. PDF 보고서는 시스템 Python으로 다음 명령을 실행한다:\n\n```bash\npython3 experiments_logs/analysis_recent/scripts/render_analysis_pdf.py latex_zips/latex_auro/asset/BRL_oracle_exp_v2/REPORT.md latex_zips/latex_auro/asset/BRL_oracle_exp_v2/02_when_what_policy_ablation/report.md\n```')
    readme.write_text(text)

    old = read(previous / '01_processed/episodes.csv')
    delta_rows = []
    for domain in ('all', 'tomato', 'wastesorting'):
        for condition in ('value_when', 'value_what'):
            def selected(rows):
                return {(r['domain'], r['scene'], r['seed']): r for r in rows
                    if r['condition'] == condition and r['status'] == 'ok'
                    and (domain == 'all' or r['domain'] == domain)}
            a, b = selected(old), selected(episodes)
            assert a.keys() == b.keys()
            wins = sum(int(b[k]['success']) > int(a[k]['success']) for k in a)
            losses = sum(int(b[k]['success']) < int(a[k]['success']) for k in a)
            row = dict(domain=domain, condition=condition, n_pairs=len(a), old_success=sum(int(x['success']) for x in a.values()),
                new_success=sum(int(x['success']) for x in b.values()), new_only_success=wins, old_only_success=losses,
                success_delta_pp=100*(wins-losses)/len(a), exact_mcnemar_p=exact_p(wins, losses),
                direction='new minus old')
            for field in ('questions', 'steps', 'seconds'):
                row[field+'_delta_mean'] = statistics.mean(float(b[k][field])-float(a[k][field]) for k in a)
            delta_rows.append(row)
    for domain in ('all', 'tomato', 'wastesorting'):
        selected_rows = sorted((r for r in delta_rows if r['domain']==domain), key=lambda r:r['exact_mcnemar_p'])
        last = 0
        for index, row in enumerate(selected_rows):
            last = max(last, min(1, (len(selected_rows)-index)*row['exact_mcnemar_p']))
            row['p_holm'] = last
    pipeline.write(PACKAGE / '01_processed/rerun_vs_previous_paired.csv', delta_rows)
    pipeline.write(PACKAGE / '01_processed/query_structure.csv', diagnostics)
    stop_counts = Counter((r['condition'], r['stop_reason']) for r in diagnostics)
    capped_low = sum(r['condition']=='value_when' and r['stop_reason']=='question_limit_reached'
        and r['final_confidence']<0.8 for r in diagnostics)
    details = ['\n## 2026-10-05 재실험과 이전 Value 배치의 paired 비교\n',
        'domain・scene・seed가 일치하는 새 실행과 이전 실행을 대응했다. 차이는 새 값 − 이전 값이다. Holm 보정은 각 domain에서 두 Value 조건에 적용했다.\n',
        '| Domain | Condition | 쌍 수 | 이전 성공 | 새 성공 | 변화(%p) | 새 실행만 성공 | 이전만 성공 | exact p | Holm p | 질문 변화(전체 평균) |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in delta_rows:
        details.append(f"| {r['domain']} | {r['condition']} | {r['n_pairs']} | {r['old_success']} | {r['new_success']} | {r['success_delta_pp']:.2f} | {r['new_only_success']} | {r['old_only_success']} | {r['exact_mcnemar_p']:.4g} | {r['p_holm']:.4g} | {r['questions_delta_mean']:.3f} |")
    details += ['\n## 실제 질문 구조 점검\n',
        f'Value-When은 질문 1개 제한 때문에 종료한 step이 {stop_counts["value_when", "question_limit_reached"]:,}개이며, 그중 답변 후 confidence가 0.8 미만인 step은 {capped_low:,}개이다. 모든 Value-When step에서 질문 수 ≤ 1을 확인했다.',
        'Value-What은 confidence를 반복 평가하며, 갱신된 belief에서 질문 Q를 다시 계산한다. 고정 1개 제한은 없다.',
        '\n| Condition | Stop reason | Steps |', '|---|---|---:|']
    details += [f'| {c} | {reason} | {n} |' for (c, reason), n in sorted(stop_counts.items())]
    details += ['\n## 설정 차이와 결과 해석\n',
        '새 Value 두 조건은 Tomato gamma=0.5, Waste gamma=0.9, query cost=0.0이다. 기본 POMCP와 가치 평가기 모두 동일 args.gamma를 사용하므로 행동 계획과 질문 가치가 함께 변했다. n_simulations의 기본값은 100이며 root 후보 수가 많으면 가치 평가 예산은 후보 수+1까지 증가할 수 있다.',
        'epsilon=0.005, max_depth=20에서 Tomato는 depth 8에 할인 종료하고 Waste는 depth 20 상한으로 종료한다. 이전 gamma=0.2는 depth 4에 할인 종료했다. 따라서 이 결과는 cost 감소, 미래 보상 가중치 변화, 탐색 범위 변화가 함께 반영된 설정 비교이다.',
        'Ours/CP는 기존 실행을 유지했다. 서로 다른 gamma와 질문 반복 규칙 때문에 이 네 조건 비교로 When/What만의 인과 효과를 주장하지 않는다. Value-When은 공통 threshold 질문 반복을 복구한 뒤 별도 재실험해야 그 설계를 평가할 수 있다.',
        '실행 cumulative reward는 할인 없는 물리 행동 보상 합이며 질문 비용을 별도 차감하지 않는다. 시간은 각 로그의 episode total_time으로, 인간 응답 시간은 포함하지 않는다.',
        '\n## 생성물\n',
        '집계 및 paired 비교는 `01_processed/`, 그림 입력은 `02_graph_data/`, PNG/PDF는 `03_figures/`에 저장했다. `paper_table.tex`는 이 패키지의 최신 요약표이며, 이전 배치 전체는 `../legacy/when_what_policy_ablation_before_20261005/`에 보존했다.']
    with (PACKAGE / 'report.md').open('a') as stream:
        stream.write('\n'.join(details)+'\n')

    with (PACKAGE / 'README.md').open('a') as stream:
        stream.write('\n## 재실험 상세\n\n최신 Value 800회와 이전 Ours/CP 800개 참조만 집계한다. 이전 Value 결과는 합산하지 않는다. `report.md`에 이전 Value와의 paired 비교 및 질문 구조 점검을 기록했다. `exp_set.md`는 기존 설정 설명과 이번 실행 설정을 함께 보존한다.\n')
    tex = [r'\begin{table}[t]', r'\centering',
        r'\caption{Policy comparison with Oracle answers. Value policies use $\gamma=0.5$ (Tomato), $0.9$ (Waste), and zero query cost; Ours and CP-When use $\gamma=0.2$. Value-When retains a one-question-per-physical-step limit. Each cell contains 200 attempted episodes. Questions and steps below use successful episodes.}',
        r'\label{tab:policy_value_rerun_20261005}', r'\begin{tabular}{llrrr}',
        r'\hline Domain & Policy & Success (\%) & Questions & Steps \\', r'\hline']
    for domain in ('tomato', 'wastesorting'):
        for condition in config['conditions']:
            r = lookup[domain, condition]
            tex.append(f"{'Tomato' if domain=='tomato' else 'Waste'} & {config['labels'][condition]} & {float(r['success_percent']):.2f} & {float(r['questions_success_only_mean']):.2f} & {float(r['steps_success_only_mean']):.2f}"+r' \\')
    tex += [r'\hline', r'\end{tabular}', r'\end{table}']
    (PACKAGE / 'paper_table.tex').write_text('\n'.join(tex)+'\n')

    # Verify all primary source hashes and independently recompute plotted means.
    for r in episodes:
        assert digest(ROOT/r['raw_source']) == r['raw_sha256']
    graph = read(PACKAGE / '02_graph_data/figure_data.csv')
    for r in graph:
        field = r['metric'].replace('_success_only', '')
        relevant = [x for x in episodes if x['domain']==r['domain'] and x['condition']==r['condition']
            and (field=='success' or x['status']=='ok')
            and (r['success_only']=='0' or x['success']=='1')
            and (field=='success' or x[field]!='')]
        values = [float(x[field]) if x['status']=='ok' else 0 for x in relevant]
        assert len(values)==int(r['n_valid'])
        assert math.isclose(statistics.mean(values)*(100 if field=='success' else 1),float(r['value']),abs_tol=1e-10)
    integrity = dict(status='passed', episodes=len(episodes), valid=sum(r['status']=='ok' for r in episodes),
        new_value_episodes=800, graph_rows=len(graph), hash_mismatch=0,
        rerun_pairs=800, value_when_max_questions_per_step=1,
        value_when_capped_below_threshold=capped_low)
    (PACKAGE/'integrity_check.json').write_text(json.dumps(integrity,indent=2)+'\n')
    shutil.copy2(previous/'exp_set.md', PACKAGE/'exp_set.md')
    settings_doc = PACKAGE/'exp_set.md'
    text = settings_doc.read_text()
    first, rest = text.split('\n', 1)
    settings_doc.write_text(first+'\n\n> 최신 실행: 2026-10-05 Value 조건은 Tomato gamma=0.5, Waste gamma=0.9, query_cost=0.0, n_simulations=100이다. 아래 1–6절은 재실험 전 점검 시점의 설정・코드 설명을 보존한 기록이며, gamma=0.2 / cost=1.0은 이전 배치 값이다. shell은 이후 새 Value 설정을 지원하도록 수정됐다. 질문 1개 제한을 포함한 실행 Python 구조는 그대로였다. 최신 결과는 report.md, 실제 설정은 마지막 절을 참조한다.\n'+rest)
    with (PACKAGE/'exp_set.md').open('a') as stream:
        stream.write('\n## 2026-10-05 실제 재실험 반영\n\n앞 절의 gamma=0.2 / query cost=1.0은 이전 배치 설정이다. 최신 Value-When・Value-What은 Tomato gamma=0.5, Waste gamma=0.9, query_cost=0.0, n_simulations=100으로 각 400회 실행했다. Ours・CP는 이전 결과를 유지한다. 질문 구조는 변경하지 않아 Value-When의 1개 제한이 그대로 적용됐다. 최신 결과와 이전 배치 paired 비교는 [report.md](report.md)를 참조한다.\n')
    if not ARCHIVE.exists():
        ARCHIVE.parent.mkdir(exist_ok=True)
        DEST.rename(ARCHIVE)
        for name in ('REPORT.md', 'REPORT.pdf', 'EXPERIMENT_SETTINGS.md', 'README.md'):
            shutil.copy2(BUNDLE/name, ARCHIVE/('bundle_previous_'+name))
    shutil.copytree(PACKAGE, DEST, dirs_exist_ok=True)
    update_bundle(lookup, delta_rows, integrity)
    print(json.dumps(integrity,indent=2))
    for r in summary:
        print(r['domain'], r['condition'], r['success_percent'], r['questions_mean'],r['questions_success_only_mean'])


def update_bundle(lookup, delta_rows, integrity):
    lines = ['## 5. When–What policy comparison (2026-10-05 Value 재실험)', '',
        '![Task success](02_when_what_policy_ablation/03_figures/success.png)', '',
        '![성공 episode 질문 수](02_when_what_policy_ablation/03_figures/questions_success_only.png)', '',
        'Ours・CP-When의 기존 결과는 유지하고 Value-When・Value-What의 각 400회를 새 배치로 교체했다. 이전 Value 결과는 합산하지 않고 legacy에 보관했다.', '',
        '| 조건 | 성공/전체 | 성공률 | 성공 episode당 질문 수 | 전체 평균 질문 수 |',
        '|---|---:|---:|---:|---:|']
    for c in ('ours', 'cp_when', 'value_when', 'value_what'):
        r = lookup['all', c]
        lines.append(f"| {c} | {r['successes']}/400 | {float(r['success_percent']):.2f}% | {float(r['questions_success_only_mean']):.2f} | {float(r['questions_mean']):.2f} |")
    lines += ['', '| 도메인 | Value-When 성공률 | Value-What 성공률 |', '|---|---:|---:|']
    for d in ('tomato','wastesorting'):
        lines.append(f"| {d} | {float(lookup[d,'value_when']['success_percent']):.2f}% | {float(lookup[d,'value_what']['success_percent']):.2f}% |")
    lines += ['', 'Value 조건의 설정은 Tomato gamma=0.5, Waste gamma=0.9, query_cost=0.0, n_simulations=100이다. Ours・CP-When은 기존 gamma=0.2 결과이다. 실제 물리 행동용 POMCP의 gamma도 함께 바뀌었다.', '',
        '이전 대비 Value-When 성공률은 60.25% → 70.00%(+9.75%p), Value-What은 98.25% → 98.75%(+0.50%p)이다. 같은 domain・scene・seed의 paired 검정에서 Value-When의 전체 개선은 Holm p=0.01063, Waste 개선은 Holm p=0.0009803이다. Tomato Value-When과 Value-What의 개선은 이 검정에서 유의하지 않았다.', '',
        'Value-What과 Ours의 paired 성공률 차이는 0.25%p, exact McNemar p=1.0이다. 성공률 동등성을 입증하는 결과는 아니며, 관측된 성공률은 비슷했다. 성공 episode 평균 질문은 Value-What 19.07회, Ours 8.49회로 약 10.58회 차이가 난다.', '',
        f"Value-When은 물리 행동당 1개 질문 제한을 유지했고, 답변 후 confidence가 0.8 미만인데 제한으로 종료한 step이 {integrity['value_when_capped_below_threshold']}개다. CP-When도 답변마다 CP gate를 재평가한다. 따라서 이 비교는 의도한 공통 threshold 질문 반복 구조의 통제 ablation이 아니며, 성능 차이를 When/What만의 효과로 해석하지 않는다.", '',
        '상세 결과, 이전 배치 paired 비교, 실제 질문 구조 점검은 [개별 report.md](02_when_what_policy_ablation/report.md)에 있다. 의도와 구현의 의사코드 비교는 [exp_set.md](02_when_what_policy_ablation/exp_set.md)를 참조한다.', '']
    report = BUNDLE/'REPORT.md'
    text = report.read_text()
    start, end = text.index('## 5. When'), text.index('## 6. Baseline')
    text = text[:start] + '\n'.join(lines) + '\n' + text[end:]
    text = text.replace('작성 기준일: 2026-09-30', '작성 기준일: 2026-10-05 (Value 정책 재실험 반영)')
    text = text.replace('4. CP/value 기반 대체 정책은 높은 성공률과 낮은 질문 수를 동시에 달성하지 못한다.',
        '4. 새 Value-What은 Ours와 비슷한 관측 성공률에서 더 많은 질문을 사용한다. Value-When은 70.0% 성공률이다. 설정과 반복 구조 차이 때문에 이 비교에서 정책만의 인과 효과는 분리하지 않는다.')
    report.write_text(text)
    settings = BUNDLE/'EXPERIMENT_SETTINGS.md'
    text = settings.read_text()
    start, end = text.index('## 5. Experiment 3:'), text.index('## 6. Experiment 4:')
    section = ['## 5. Experiment 3: When–What policy comparison', '',
        '2026-10-05 Value-When・Value-What을 각 400회 재실행했다. Ours・CP-When은 기존 각 400회로 유지한다. 정상 결과 1,596개와 CP 오류 4개를 포함한다.', '',
        '| 조건 | When | What | 반복·종료 | Tomato gamma | Waste gamma | query cost | simulations |',
        '|---|---|---|---|---:|---:|---:|---:|',
        '| Ours | confidence < 0.8 | EIG | confidence threshold | 0.2 | 0.2 | 1.0 (value 평가 미사용) | 100 |',
        '| CP-When | action CP ambiguity | EIG | 답변마다 CP 재평가 | 0.2 | 0.2 | 1.0 (value 평가 미사용) | 100 |',
        '| Value-When | max Q(query) > max Q(next physical) | EIG | 물리 행동당 최대 1개 | 0.5 | 0.9 | 0.0 | 100 |',
        '| Value-What | confidence < 0.8 | 최고 질문 Q | confidence threshold | 0.5 | 0.9 | 0.0 | 100 |', '',
        '공통 max_step=50, max_depth=20, epsilon=0.005, UCB c=1.0이다. Value 평가의 failure_penalty=10.0, answer_accuracy=1.0이며 답변은 auto Oracle이다. CP의 qhat은 Tomato 0.8404, Waste 0.8704, score_temperature=5.0이다.', '',
        'Value gamma는 질문 평가와 실제 물리 행동 계획 모두에 적용한다. 가치 평가 simulation 예산은 max(100, root 후보 수+1)로 늘어날 수 있다. query cost 0은 질문 보상을 0으로 만든다.', '',
        'When은 관측으로 belief를 갱신한 다음, 다음 물리 행동 전에 판단한다. 질문 시작을 판단한 최고 Q 질문을 Value-When에서 실제로 그대로 묻는 것은 아니며 What은 EIG이다. Value-What은 root에서 질문 Q끼리 비교하고, 물리 행동 Q와 비교하여 질문을 취소하지 않는다.', '',
        '**설계 제한:** Value-When의 1개 제한 및 CP의 반복 gate 평가가 남아 있다. 공통 threshold 반복을 유지하며 When만 교체한다는 의도와 불일치하므로, 현재 결과는 설정 및 구현 비교로 보고한다.', '',
        '| 조건 | 성공률 | 전체 평균 질문 | 성공 실행 평균 질문 |', '|---|---:|---:|---:|']
    for c in ('ours','cp_when','value_when','value_what'):
        r=lookup['all',c]
        section.append(f"| {c} | {float(r['success_percent']):.2f}% | {float(r['questions_mean']):.3f} | {float(r['questions_success_only_mean']):.3f} |")
    section += ['', '새 배치: `experiments_logs/when_what_policy_ablation/20261005_190346_519071`. 이전 Value 결과는 `legacy/when_what_policy_ablation_before_20261005/`에 보존한다. 세부 해석과 paired 검정은 개별 `report.md`를 참조한다.', '']
    text=text[:start]+'\n'.join(section)+'\n'+text[end:]
    settings.write_text(text.replace('작성 기준일: 2026-09-30','작성 기준일: 2026-10-05 (Value 정책 재실험 반영)'))
    readme=BUNDLE/'README.md'
    text=readme.read_text()
    marker='\n## 2026-10-05 Value 재실험 반영\n'
    if marker in text:
        text=text.split(marker)[0]
    text+=marker+'\n`02_when_what_policy_ablation/`의 Value-When・Value-What은 새 800회 결과로 교체했다. 이전 결과는 `legacy/when_what_policy_ablation_before_20261005/`에 보존했다. Tomato gamma=0.5, Waste gamma=0.9, query_cost=0.0이며 Value-When의 1개 질문 제한은 유지됐다. 설계 제한과 이전 배치 비교는 개별 보고서에 명시했다.\n'
    readme.write_text(text)
    check=BUNDLE/'raw_integrity_check.json'
    data=json.loads(check.read_text())
    data['policy_rerun_20261005']=integrity
    data['policy_rerun_20261005']['package']='02_when_what_policy_ablation'
    data['policy_rerun_20261005']['previous_validation']='legacy/when_what_policy_ablation_before_20261005'
    check.write_text(json.dumps(data,indent=2)+'\n')


if __name__ == '__main__':
    main()
