"""Publish the 400-episode Value-When rerun after removing its question cap."""
from collections import Counter
from pathlib import Path
import json
import math
import shutil
import statistics
import subprocess

import pipeline
from refresh_policy_ablation_20261005 import read, digest, exact_p

ROOT = pipeline.PROJECT
BUNDLE = ROOT / 'latex_zips/latex_auro/asset/BRL_oracle_exp_v2'
DEST = BUNDLE / '02_when_what_policy_ablation'
PREVIOUS = pipeline.BASE / 'when_what_policy_ablation_20261005'
PACKAGE = pipeline.BASE / 'when_what_policy_ablation_20261005_threshold_loop'
RUN = ROOT / 'experiments_logs/when_what_policy_ablation/20261005_204332_446690'
ARCHIVE = BUNDLE / 'legacy/when_what_policy_ablation_before_threshold_loop'


def main():
    old_config = json.loads((PREVIOUS/'manifest.json').read_text())
    runs = read(RUN/'runs.csv')
    assert len(runs)==400 and all(r['status']=='complete' and r['condition']=='value_when' for r in runs)
    assert Counter((r['domain'], int(r['scene'])) for r in runs)==Counter({
        (d,s):40 for d in ('tomato','wastesorting') for s in range(1,6)})
    PACKAGE.mkdir(exist_ok=True)
    sources=[]
    for item in old_config['sources']:
        if item['condition']=='value_when':
            continue
        original=ROOT/item['source_file']
        assert digest(original)==item['sha256']
        relative=Path(item['source_file'].split('/00_raw/',1)[1])
        target=PACKAGE/'00_raw'/relative
        shutil.copytree(original.parent,target.parent,dirs_exist_ok=True)
        sources.append(dict(item,source_file=str(target.relative_to(ROOT))))
    traces={}
    diagnostics=[]
    for row in runs:
        folder=Path(row['log_dir'])
        logs=list(folder.glob('exp_*.txt'))
        assert len(logs)==1
        data=json.loads((folder/'policy_trace.json').read_text())
        meta=data['meta']
        assert meta['condition']=='value_when' and meta['domain']==row['domain'] and meta['seed']==int(row['seed'])
        assert meta['gamma']==(0.5 if row['domain']=='tomato' else 0.9)
        assert meta['query_cost']==0 and meta['n_simulations']==100 and meta['threshold']==0.8
        target=PACKAGE/'00_raw'/folder.relative_to(RUN)
        shutil.copytree(folder,target,dirs_exist_ok=True)
        log=target/logs[0].name
        sources.append(dict(source_file=str(log.relative_to(ROOT)),sha256=digest(log),domain=row['domain'],
            condition='value_when',scene=f"{int(row['scene']):02}",iteration=row['iteration'],seed=row['seed'],exit_code='0'))
        traces[row['domain'],str(int(row['scene'])),row['seed']]=data
        for index,step in enumerate(data['policy_traces'],1):
            assert len(step['when_decisions'])==1
            assert step['stop_reason']!='question_limit_reached'
            if step['stop_reason']=='threshold_reached':
                assert step['final_confidence']>=0.8
            assert len({q['question'] for q in step['questions']})==len(step['questions'])
            for question in step['questions'][1:]:
                assert question['confidence_before']<0.8
            diagnostics.append(dict(domain=row['domain'],scene=row['scene'],seed=row['seed'],step=index,
                questions=len(step['questions']),stop_reason=step['stop_reason'],
                final_confidence=step['final_confidence'],triggered=step['when']['start']))

    notes=('Value-When은 2026-10-05 20:43 배치 400회로 교체했다. 질문 1개 제한을 제거하고 '
        'Q 기반 When은 episode 진입 시 한 번만 평가하며, 진입 후 confidence threshold=0.8로 반복・종료한다. '
        'Value-What은 같은 날 19:03 배치 400회 결과를 유지한다. 두 Value 조건은 Tomato gamma=0.5, Waste gamma=0.9, '
        'query_cost=0.0, n_simulations=100이다. Ours와 CP-When은 기존 각 400회를 유지하고 gamma=0.2이다. '
        '보관된 CP-When 결과는 수정 전 답변마다 CP 재평가 구현의 결과이므로, 현재 수정된 CP 코드의 평가로 해석하지 않는다. '
        '기존 CP parsing 오류 4건은 성공률에서 실패로 집계한다. 이전 Value-When과 최신 배치는 합산하지 않는다.')
    config=dict(old_config,sources=sources,title='When–What policy comparison (Value-When threshold-loop rerun)',
        notes=notes,status='analyzed_with_reference_batch_differences',key_findings=[],
        status_note='수정된 Value-When 400/400 정상 완료. 기존 Value-What 400회 및 Ours/CP 800개 참조 유지.',
        rerun_root=str(RUN.relative_to(ROOT)),reference_batch='when_what_policy_ablation_20261005',
        previous_published_package=str(ARCHIVE.relative_to(ROOT)))
    manifest=PACKAGE/'manifest.json'
    manifest.write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n')
    prov=PACKAGE/'provenance'
    prov.mkdir(exist_ok=True)
    shutil.copy2(RUN/'manifest.json',prov/'value_when_run_manifest.json')
    shutil.copy2(RUN/'runs.csv',prov/'value_when_runs.csv')
    shutil.copytree(PREVIOUS/'provenance',prov/'previous_parameter_rerun',dirs_exist_ok=True)
    for name in ('policies.py','query_episode.py','runner.py','value_evaluator.py','test_policy_ablation.py'):
        shutil.copy2(ROOT/'scripts/ablation/when_what_policy_ablation/script'/name,prov/name)
    shutil.copy2(Path(__file__),prov/Path(__file__).name)
    (prov/'README.md').write_text('Source snapshots were captured during analysis, not automatically at launch. Executed parameters and question structure are verified against raw policy traces.\n')
    pipeline.stage1(PACKAGE)
    episodes=read(PACKAGE/'01_processed/episodes.csv')
    for r in episodes:
        if r['condition']=='value_when':
            trace=traces[r['domain'],str(int(r['scene'])),r['seed']]
            assert r['status']=='ok' and int(r['success'])==int(trace['success'])
            assert int(r['questions'])==trace['total_questions']
            assert int(r['steps'])==len(trace['policy_traces'])
    pipeline.stage2(PACKAGE)
    summary=read(PACKAGE/'01_processed/summary.csv')
    lookup={(r['domain'],r['condition']):r for r in summary}
    when=lookup['all','value_when']
    config['key_findings']=[
        f"수정된 Value-When 성공률 {float(when['success_percent']):.2f}% ({when['successes']}/400). 전체 평균 질문 {float(when['questions_mean']):.3f}회, 성공 실행 평균 {float(when['questions_success_only_mean']):.3f}회.",
        '같은 gamma/cost에서 1개 제한을 사용한 이전 Value-When 70.00%보다 13.75%p 높아졌다. Tomato는 65.50% → 68.50%, Waste는 74.50% → 99.00%이다.',
        'Value-What은 재실행하지 않고 기존 395/400(98.75%) 결과를 유지했다.',
        '모든 새 Value-When step에서 When을 1회 평가했으며, 질문 1개 제한에 의한 종료는 없다.']
    manifest.write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n')
    pipeline.stage2(PACKAGE)
    pipeline.stage3(PACKAGE)
    pipeline.docs(PACKAGE)

    old=read(PREVIOUS/'01_processed/episodes.csv')
    deltas=[]
    for domain in ('all','tomato','wastesorting'):
        def selected(rows):
            return {(r['domain'],r['scene'],r['seed']):r for r in rows if r['condition']=='value_when'
                and r['status']=='ok' and (domain=='all' or r['domain']==domain)}
        a,b=selected(old),selected(episodes)
        assert a.keys()==b.keys()
        wins=sum(int(b[k]['success'])>int(a[k]['success']) for k in a)
        losses=sum(int(b[k]['success'])<int(a[k]['success']) for k in a)
        delta=dict(domain=domain,n_pairs=len(a),old_success=sum(int(r['success']) for r in a.values()),
            new_success=sum(int(r['success']) for r in b.values()),new_only_success=wins,old_only_success=losses,
            success_delta_pp=100*(wins-losses)/len(a),exact_mcnemar_p=exact_p(wins,losses),direction='new minus old')
        for field in ('questions','steps','seconds'):
            delta[field+'_delta_mean']=statistics.mean(float(b[k][field])-float(a[k][field]) for k in a)
        deltas.append(delta)
    pipeline.write(PACKAGE/'01_processed/threshold_loop_vs_one_question_paired.csv',deltas)
    pipeline.write(PACKAGE/'01_processed/query_structure.csv',diagnostics)
    counts=Counter((r['domain'],r['stop_reason']) for r in diagnostics)
    multi=Counter(r['domain'] for r in diagnostics if r['questions']>1)
    maximum={d:max(r['questions'] for r in diagnostics if r['domain']==d) for d in ('tomato','wastesorting')}
    extra=['\n## 질문 1개 제한 제거 전후 paired 비교\n',
        '같은 gamma/cost, domain・scene・seed의 400쌍을 대응했다. 차이는 새 값 − 이전 값이다. 각 domain에서 Value-When 1개 비교의 양측 exact McNemar p를 보고한다. 전체와 domain 검정을 중복 독립 증거로 해석하지 않는다.\n',
        '| Domain | 이전 성공 | 새 성공 | 변화(%p) | 새 실행만 성공 | 이전만 성공 | exact p | 전체 평균 질문 변화 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in deltas:
        extra.append(f"| {r['domain']} | {r['old_success']}/{r['n_pairs']} | {r['new_success']}/{r['n_pairs']} | {r['success_delta_pp']:.2f} | {r['new_only_success']} | {r['old_only_success']} | {r['exact_mcnemar_p']:.5g} | {r['questions_delta_mean']:.3f} |")
    extra+=['\n## 새 Value-When의 실제 질문 구조\n',
        'When은 관측 belief 갱신 후 다음 물리 행동 전에 한 번 판단한다. 시작 후 EIG로 질문을 선택하고, 답변 후 confidence < 0.8이면 추가 질문한다. 후보 없음・belief 미감소도 종료 조건이다. 같은 episode에서 같은 사실을 반복하지 않는다.',
        '\n| Domain | 2개 이상 질문한 steps | step당 최대 질문 |', '|---|---:|---:|']
    extra += [f'| {d} | {multi[d]} | {maximum[d]} |' for d in maximum]
    extra+=['\n| Domain | 종료 이유 | Steps |', '|---|---|---:|']
    extra += [f'| {d} | {reason} | {n} |' for (d,reason),n in sorted(counts.items())]
    extra+=['\n## 유지한 비교군과 해석 범위\n',
        'Value-What의 raw source와 SHA-256은 이전 19:03 배치 400개와 모두 동일하다. Ours・CP 참조 결과도 그대로다. CP의 보관된 결과는 현재 공통 threshold 반복으로 수정된 코드의 결과가 아니다. 최신 네 조건은 gamma와 CP 반복 구현이 다르므로 정책만의 완전한 통제 비교로 주장하지 않는다.',
        'Value-When 전후 비교는 같은 gamma/cost에서 질문 1개 제한과 질문 반복 구조 변경을 평가한다. 초기 seed가 같아도 질문・행동 경로가 달라지면 난수 소비가 달라진다. 성공 실행 질문 평균과 전체 평균을 구분해 보고한다.',
        'cumulative reward는 할인 없는 물리 행동 보상 합이고 질문 비용을 별도 차감하지 않는다. 새 배치의 총 성공률과 실패 원인은 task 수준 결과이며 실행 오류는 0건이다.',
        '\n## 출처 및 재생성\n',f'새 실행: `{RUN.relative_to(ROOT)}`.',
        '원본 SHA-256, episode CSV와 trace, 그림 집계 일치는 `integrity_check.json`에 기록했다.',
        '```bash','/home/fr/miniconda3/envs/brl/bin/python experiments_logs/analysis_recent/scripts/refresh_policy_ablation_threshold_loop.py','```']
    with (PACKAGE/'report.md').open('a') as f:f.write('\n'.join(extra)+'\n')
    make_table(lookup)

    graph=read(PACKAGE/'02_graph_data/figure_data.csv')
    for r in episodes:assert digest(ROOT/r['raw_source'])==r['raw_sha256']
    for r in graph:
        field=r['metric'].replace('_success_only','')
        rows=[x for x in episodes if x['condition']==r['condition'] and x['domain']==r['domain']
            and (field=='success' or x['status']=='ok') and (r['success_only']=='0' or x['success']=='1')]
        values=[float(x[field]) if x['status']=='ok' else 0 for x in rows]
        assert len(values)==int(r['n_valid'])
        assert math.isclose(statistics.mean(values)*(100 if field=='success' else 1),float(r['value']),abs_tol=1e-10)
    old_what={(r['domain'],r['scene'],r['seed']):r['raw_sha256'] for r in old if r['condition']=='value_what'}
    new_what={(r['domain'],r['scene'],r['seed']):r['raw_sha256'] for r in episodes if r['condition']=='value_what'}
    assert old_what==new_what and len(new_what)==400
    integrity=dict(status='passed',episodes=1600,valid=1596,new_value_when_episodes=400,
        value_what_unchanged=400,hash_mismatch=0,graph_rows=len(graph),paired_rerun_episodes=400,
        value_when_when_calls_per_step=1,question_limit_reached=0,multi_question_steps=dict(multi),
        max_questions_per_step=maximum)
    (PACKAGE/'integrity_check.json').write_text(json.dumps(integrity,indent=2)+'\n')
    doc=PREVIOUS/'exp_set.md'
    text=doc.read_text()
    text=text.replace('> 최신 실행: 2026-10-05 Value 조건은', '> 직전 19:03 배치의 상태 기록: Value 조건은')
    text=text.replace('## 2026-10-05 실제 재실험 반영', '## 2026-10-05 19:03 배치 이력')
    header,rest=text.split('\n',1)
    text=header+'\n\n> 최신 상태: Value-When은 1개 제한 제거 후 2026-10-05 20:43 배치 400회 결과로 교체했다. 질문 시작 후 공통 threshold 0.8에 따라 반복한다. 아래 이전 구현 설명은 이력이며 현재 코드와 구분한다. Value-What은 19:03 배치 400회를 유지한다. 상세 결과는 report.md를 참조한다.\n'+rest
    text+='\n## 2026-10-05 20:43 최신 Value-When 실행\n\nValue-When의 1개 제한을 제거했다. When은 질문 episode 진입 시 한 번 평가하고, 답변 후 confidence가 0.8 미만이면 후보가 남고 belief가 감소한 경우 EIG 질문을 계속한다. confidence가 0.8 이상이거나 후보가 없거나 belief가 감소하지 않으면 종료한다. 같은 fact를 반복하지 않는다. 실제 400회 trace에서 이 구조를 확인했다. CP 코드도 이 공통 반복으로 수정됐지만, 현재 보관된 CP 실험 결과는 수정 전 코드의 결과이다. Value-What은 원래 같은 threshold 반복이므로 19:03 결과를 그대로 유지한다.\n\nTomato gamma=0.5, Waste gamma=0.9, query_cost=0.0, n_simulations=100이다. 최신 Value-When 성공률은 335/400(83.75%)이다.\n'
    (PACKAGE/'exp_set.md').write_text(text)
    (PACKAGE/'README.md').write_text((PACKAGE/'README.md').read_text().replace(
        f'python3 experiments_logs/analysis_recent/scripts/pipeline.py --package {PACKAGE.name} --stage all',
        '/home/fr/miniconda3/envs/brl/bin/python experiments_logs/analysis_recent/scripts/refresh_policy_ablation_threshold_loop.py')+
        '\n## 최신 배치\n\nValue-When만 20:43 배치로 교체했다. Value-What과 Ours/CP 참조는 유지한다. 이전 1개 제한 배치와 합산하지 않는다. 상세 전후 비교와 질문 구조는 report.md에 기록했다.\n')
    if not ARCHIVE.exists():
        DEST.rename(ARCHIVE)
        for name in ('REPORT.md','REPORT.pdf','README.md','EXPERIMENT_SETTINGS.md'):
            shutil.copy2(BUNDLE/name,ARCHIVE/('bundle_previous_'+name))
    shutil.copytree(PACKAGE,DEST,dirs_exist_ok=True)
    update_bundle(lookup,deltas,integrity)
    subprocess.run(['python3',str(pipeline.BASE/'scripts/render_analysis_pdf.py'),str(BUNDLE/'REPORT.md'),str(DEST/'report.md')],check=True)
    print(json.dumps(integrity,indent=2))
    for r in summary:print(r['domain'],r['condition'],r['success_percent'],r['questions_mean'],r['questions_success_only_mean'])


def make_table(lookup):
    lines=[r'\begin{table}[t]',r'\centering',
        r'\caption{Oracle policy comparison. Value-When uses the corrected confidence-threshold question loop. Value policies use $\gamma=0.5$ (Tomato), $0.9$ (Waste), and zero query cost; Ours and archived CP-When use $\gamma=0.2$. Each cell contains 200 attempted episodes. Questions and steps use successful episodes. Archived CP-When used its previous repeated-CP gate.}',
        r'\label{tab:policy_value_threshold_loop}',r'\begin{tabular}{llrrr}',
        r'\hline Domain & Policy & Success (\%) & Questions & Steps \\',r'\hline']
    for d in ('tomato','wastesorting'):
        for c in ('ours','cp_when','value_when','value_what'):
            r=lookup[d,c]
            lines.append(f"{'Tomato' if d=='tomato' else 'Waste'} & {c.replace('_','-')} & {float(r['success_percent']):.2f} & {float(r['questions_success_only_mean']):.2f} & {float(r['steps_success_only_mean']):.2f}"+r' \\')
    lines += [r'\hline',r'\end{tabular}',r'\end{table}']
    (PACKAGE/'paper_table.tex').write_text('\n'.join(lines)+'\n')


def update_bundle(lookup,deltas,integrity):
    section=['## 5. When–What policy comparison (Value-When 반복 구조 수정 후)', '',
        '![Task success](02_when_what_policy_ablation/03_figures/success.png)', '',
        '![성공 실행 질문 수](02_when_what_policy_ablation/03_figures/questions_success_only.png)', '',
        'Value-When만 2026-10-05 20:43 배치 400회로 교체했다. Value-What은 앞서 완료한 19:03 배치, Ours・CP-When은 기존 참조 결과를 유지한다.', '',
        '| 조건 | 성공/전체 | 성공률 | 성공 실행 평균 질문 | 전체 평균 질문 |', '|---|---:|---:|---:|---:|']
    for c in ('ours','cp_when','value_when','value_what'):
        r=lookup['all',c]
        section.append(f"| {c} | {r['successes']}/400 | {float(r['success_percent']):.2f}% | {float(r['questions_success_only_mean']):.2f} | {float(r['questions_mean']):.2f} |")
    section += ['', '| Value-When domain | 1개 제한 이전 | threshold 반복 이후 |', '|---|---:|---:|',
        '| Tomato | 65.50% | 68.50% |','| Waste | 74.50% | 99.00% |', '',
        'Value-When의 전체 성공률은 70.00% → 83.75%(+13.75%p)이다. 같은 gamma/cost에서 질문 시작 이후 confidence threshold에 따라 여러 번 묻도록 수정한 결과이다. Value 두 조건의 gamma는 Tomato 0.5, Waste 0.9, query_cost=0.0, n_simulations=100이다.', '',
        f"전후 paired exact McNemar p는 전체 {deltas[0]['exact_mcnemar_p']:.5g}, Tomato {deltas[1]['exact_mcnemar_p']:.5g}, Waste {deltas[2]['exact_mcnemar_p']:.5g}이다. 전체와 domain 검정은 중복 독립 증거가 아니다.", '',
        f"실제 로그에서 한 step에 여러 번 질문한 경우는 Tomato {integrity['multi_question_steps'].get('tomato',0)}개, Waste {integrity['multi_question_steps'].get('wastesorting',0)}개이다. step당 최대 질문은 각각 7개, 10개였고 질문 제한 종료는 0건이다.", '',
        'Ours・CP는 gamma=0.2 참조 결과이며, 보관된 CP는 수정 전 답변마다 CP를 재평가한 구현이다. 현재 수정된 CP 코드의 결과로 해석하지 않는다. 최신 네 조건은 설정 차이가 있어 정책만의 완전한 통제 비교로 주장하지 않는다.', '',
        '상세 paired 비교와 질문 구조 점검은 [report.md](02_when_what_policy_ablation/report.md), 설정 이력은 [exp_set.md](02_when_what_policy_ablation/exp_set.md)에 있다.', '']
    report=BUNDLE/'REPORT.md'
    text=report.read_text();a,b=text.index('## 5. When'),text.index('## 6. Baseline')
    text=text[:a]+'\n'.join(section)+'\n'+text[b:]
    text=text.replace('Value-When은 70.0% 성공률이다.','수정된 Value-When은 83.75% 성공률이다.')
    report.write_text(text)
    settings=BUNDLE/'EXPERIMENT_SETTINGS.md'
    text=settings.read_text()
    text=text.replace('물리 행동당 최대 1개 | 0.5 | 0.9','진입 후 confidence threshold 반복 | 0.5 | 0.9')
    a,b=text.index('## 5. Experiment 3:'),text.index('## 6. Experiment 4:')
    section=text[a:b]
    section=section.replace('2026-10-05 Value-When・Value-What을 각 400회 재실행했다.',
        'Value-When은 2026-10-05 20:43 질문 반복 수정 배치 400회, Value-What은 19:03 배치 400회로 구성한다.')
    label='**설계 제한:**' if '**설계 제한:**' in section else '**참조 결과 제한:**'
    start=section.index(label);end=section.index('\n\n',start)
    section=section[:start]+'**참조 결과 제한:** 새 Value-When에는 질문 1개 제한이 없다. CP의 보관된 결과는 수정 전 반복 gate 구현이다. Ours/CP와 Value의 gamma도 달라 네 조건 전체를 정책만의 통제 비교로 해석하지 않는다.'+section[end:]
    start=section.index('| 조건 | 성공률 |');end=section.index('\n\n',start)
    table=['| 조건 | 성공률 | 전체 평균 질문 | 성공 실행 평균 질문 |','|---|---:|---:|---:|']
    for c in ('ours','cp_when','value_when','value_what'):
        r=lookup['all',c];table.append(f"| {c} | {float(r['success_percent']):.2f}% | {float(r['questions_mean']):.3f} | {float(r['questions_success_only_mean']):.3f} |")
    section=section[:start]+'\n'.join(table)+section[end:]
    section=section.replace('새 배치: `experiments_logs/when_what_policy_ablation/20261005_190346_519071`.',
        'Value-What 유지 배치: `experiments_logs/when_what_policy_ablation/20261005_190346_519071`.')
    marker='\nValue-When 새 배치:'
    if marker in section:section=section.split(marker)[0]
    section+='\nValue-When 새 배치: `experiments_logs/when_what_policy_ablation/20261005_204332_446690`. 1개 제한을 사용한 직전 패키지는 `legacy/when_what_policy_ablation_before_threshold_loop/`에 보존한다.\n\n'
    settings.write_text(text[:a]+section+text[b:])
    readme=BUNDLE/'README.md';text=readme.read_text()
    marker='\n## 2026-10-05 Value 재실험 반영\n'
    if marker in text:text=text.split(marker)[0]
    text+=marker+'\n최신 Value-When은 20:43 질문 1개 제한 제거 배치 400회이다. Value-What은 19:03 배치 400회를 유지한다. 반복 구조 수정 전 패키지는 `legacy/when_what_policy_ablation_before_threshold_loop/`에 보존한다. 상세 분석은 개별 보고서를 참조한다.\n'
    readme.write_text(text)
    check=BUNDLE/'raw_integrity_check.json';data=json.loads(check.read_text())
    data['policy_threshold_loop_rerun_20261005']=integrity
    check.write_text(json.dumps(data,indent=2)+'\n')


if __name__=='__main__':
    main()
