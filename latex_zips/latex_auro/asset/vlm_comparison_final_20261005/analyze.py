"""Freeze the selected 160 VLM runs and generate final descriptive analyses."""
from collections import Counter
from pathlib import Path
import csv
import hashlib
import json
import math
import re
import shutil
import statistics as st
import subprocess
import sys

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
PREVIOUS=ROOT/'collect_exp/vlm_comparison_20261005'
LATEST=[
    '20261005_203307_gui_knowno_scene1','20261005_203728_gui','20261005_204002_gui',
    '20261005_204423_gui_knowno_scene2','20261005_204921_gui','20261005_205328_gui',
    '20261005_205725_gui','20261005_210208_gui','20261005_210358_gui',
    '20261005_210653_gui','20261005_211145_gui','20261005_211523_gui','20261005_212154_gui',
]
METRICS=['questions','query_probability','steps','total_time_s','interaction_time_s',
         'execute_time_s','search_time_s','planning_residual_time_s']


def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def write(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def value(text,key,default=None):
    match=re.search(r'^'+re.escape(key)+r': (.*)$',text,re.M)
    if match:return match[1]
    if default is not None:return default
    raise ValueError('Missing field '+key)


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def wilson(k,n):
    z=1.959963984540054;p=k/n;den=1+z*z/n
    mid=(p+z*z/(2*n))/den;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return 100*max(0,mid-half),100*min(1,mid+half)


def fisher(a,b):
    successes=a+b;n=40;den=math.comb(80,40)
    weights={k:math.comb(successes,k)*math.comb(80-successes,n-k)/den
        for k in range(max(0,successes-n),min(n,successes)+1)}
    assert math.isclose(sum(weights.values()),1,abs_tol=1e-10)
    return min(1,sum(p for p in weights.values() if p<=weights[a]*(1+1e-12)))


def parse(source):
    path=ROOT/source['source_file'];text=path.read_text()
    domain=source['domain'];method=source['method'];scene=int(source['scene'])
    folder=path.parent.parent
    recorded_method=value(text,'planner','')
    if recorded_method:
        assert recorded_method==method
        method_source='Meta.planner'
    else:
        assert method+'_planner_node.py' in (folder/'planner.log').read_text(errors='replace')
        method_source='planner.log command'
    assert value(text,'domain')==('tomato' if domain=='tomato' else 'wastesorting')
    scenario=value(text,'scenario','')
    if scenario:assert scenario==('T' if domain=='tomato' else 'W')+str(scene)
    feedback=folder/'feedback_manager.log'
    fb=feedback.read_text(errors='replace')
    assert re.search(r'_use_vlm:=true|use_vlm=True',fb),folder
    steps=int(value(text,'steps'));questions=int(value(text,'total_questions'))
    section=re.search(r'^\[Questions\]\n(.*?)(?=^\[|\Z)',text,re.M|re.S)[1]
    question_steps=[int(m[1]) for m in re.finditer(r'^Step=(\d+):\n(.*?)(?=^Step=|\Z)',section,re.M|re.S)
        if re.search(r'^- Q\d+:',m[2],re.M)]
    assert questions==len(re.findall(r'^- Q\d+:',section,re.M))
    assert steps==len(re.findall(r'^STEP \d+:',text,re.M)) and steps>0
    assert len(question_steps)==len(set(question_steps))
    raw=OUT/'00_raw'/domain/source['run']
    raw.mkdir(parents=True,exist_ok=True)
    copied=[]
    for original in (path,feedback,folder/'planner.log'):
        if original.is_file():
            target=raw/original.name;shutil.copy2(original,target)
            assert sha(original)==sha(target)
            copied.append(dict(source=str(original.relative_to(ROOT)),copy=str(target.relative_to(OUT)),sha256=sha(original)))
    row=dict(domain=domain,method=method,method_source=method_source,scene=scene,run=source['run'],scene_source=source['scene_source'],
        success=value(text,'success')=='True',end_reason=value(text,'end_reason'),steps=steps,
        questions=questions,query_steps=len(question_steps),query_probability=len(question_steps)/steps,
        total_time_s=float(value(text,'total_time').rstrip('s')),source_file=source['source_file'],
        raw_copy=str((raw/path.name).relative_to(OUT)),raw_sha256=sha(path))
    for metric in ('interaction_time','execute_time','search_time'):
        row[metric+'_s']=float(re.search(r'^'+metric+r': total=([\d.]+)s',text,re.M)[1])
    row['planning_residual_time_s']=row['total_time_s']-row['interaction_time_s']-row['execute_time_s']
    assert row['planning_residual_time_s']>=-0.0002
    for metric in METRICS:assert math.isfinite(row[metric])
    for key in ('seed','model','qhat','prompt_version','generation_temperature','score_temperature','gamma','n_simulations'):
        row[key]=value(text,key,'')
    row['failure_category']=''
    if not row['success']:
        reason=row['end_reason']
        if reason=='no user selection' and 'VLM did not return a candidate token' in fb:
            row['failure_category']='vlm_candidate_token_parse_failure'
        elif 'precondition failure' in reason:row['failure_category']='precondition_failure'
        elif 'Expert' in reason or 'expert' in reason:row['failure_category']='expert_reported_failure'
        else:row['failure_category']=reason
    return row,copied


def main():
    selected=[]
    for domain in ('tomato','waste'):
        for row in read(PREVIOUS/f'{domain}_vlm_interim_20261005/runs.csv'):
            selected.append(dict(domain=domain,method=row['method'],scene=row['scene'],run=row['run'],
                scene_source=row['scene_source'],source_file=row['source_file']))
    assert len(selected)==147
    for i,name in enumerate(LATEST):
        folder=ROOT/'collect_exp/logs/04_BRL_WASTE/logs/master'/name
        paths=list(folder.glob('*experiments/*.txt'));assert len(paths)==1
        selected.append(dict(domain='waste',method='knowno',scene=1 if i<3 else 2,run=name,
            scene_source='Meta.scenario; user-designated final batch',source_file=str(paths[0].relative_to(ROOT))))
    assert len(selected)==160 and len({(r['domain'],r['run']) for r in selected})==160
    expected={(d,m,s):10 for d in ('tomato','waste') for m in ('pomdp','knowno') for s in range(1,5)}
    assert Counter((r['domain'],r['method'],int(r['scene'])) for r in selected)==Counter(expected)
    write(OUT/'selected_manifest.csv',selected)
    rows=[];copies=[]
    for source in selected:
        row,files=parse(source);rows.append(row);copies+=files
    write(OUT/'runs.csv',rows);write(OUT/'files_sha256.csv',copies)
    write(OUT/'failures.csv',[r for r in rows if not r['success']])
    summary=[]
    for domain in ('tomato','waste'):
        dest=OUT/f'{domain}_vlm_final_20261005';dest.mkdir(exist_ok=True)
        write(dest/'runs.csv',[r for r in rows if r['domain']==domain])
        write(dest/'failures.csv',[r for r in rows if r['domain']==domain and not r['success']])
        for method in ('pomdp','knowno'):
            for scene in (1,2,3,4,'all'):
                group=[r for r in rows if r['domain']==domain and r['method']==method and (scene=='all' or r['scene']==scene)]
                success=[r for r in group if r['success']];lo,hi=wilson(len(success),len(group))
                result=dict(domain=domain,method=method,scene=scene,n=len(group),successes=len(success),
                    failures=len(group)-len(success),success_percent=100*len(success)/len(group),ci95_low=lo,ci95_high=hi)
                for metric in METRICS:
                    for label,subset in (('all',group),('success',success)):
                        values=[r[metric] for r in subset]
                        result[metric+'_mean_'+label]=st.mean(values) if values else ''
                        result[metric+'_sd_'+label]=st.stdev(values) if len(values)>1 else ''
                summary.append(result)
        write(dest/'summary.csv',[r for r in summary if r['domain']==domain])
    write(OUT/'summary.csv',summary)
    progress=[dict(domain=r['domain'],scene=r['scene'],planner=r['method'],expected=10,result_count=r['n'],difference=r['n']-10,reported_successes=r['successes']) for r in summary if r['scene']!='all']
    write(OUT/'progress.csv',progress)
    lookup={(r['domain'],r['method'],r['scene']):r for r in summary}
    tests=[dict(domain=d,pomdp_success=lookup[d,'pomdp','all']['successes'],knowno_success=lookup[d,'knowno','all']['successes'],
        success_delta_pp=lookup[d,'pomdp','all']['success_percent']-lookup[d,'knowno','all']['success_percent'],
        fisher_exact_p=fisher(lookup[d,'pomdp','all']['successes'],lookup[d,'knowno','all']['successes'])) for d in ('tomato','waste')]
    last=0
    for i,r in enumerate(sorted(tests,key=lambda r:r['fisher_exact_p'])):
        last=max(last,min(1,(2-i)*r['fisher_exact_p']));r['holm_p']=last
    write(OUT/'success_comparisons.csv',tests)
    (OUT/'analysis_definitions.json').write_text(json.dumps(dict(date='2026-10-05 Asia/Seoul',
        selection='Previous 147 selected current runs plus exactly 13 user-listed Waste KnowNo runs; 160 total.',
        balance='2 domains x 2 methods x 4 scenes x 10 runs.',
        success='Planner-reported; not independent physical/video verification.',
        other_metrics='Separate successful-only and all-run arithmetic means; sample SD ddof=1.',
        query_probability='Mean of each run\'s distinct question-bearing steps / logged executed steps.',
        timing='Planner Timing.total_time, not GUI wall time; interaction includes VLM feedback processing, not human response time.',
        residual='total_time - interaction_time - execute_time; includes search/update/pruning/overhead.',
        exclusions='Earlier VLM pilot runs, previously excluded human session, and incomplete 20261002_175118_gui are not added.',
        comparisons='Unpaired two-sided Fisher exact per domain, Holm correction for two tests; no matched-seed pairing is assumed.'),ensure_ascii=False,indent=2)+'\n')
    report(rows,lookup,tests)
    tex=[r'\begin{table}[t]',r'\centering',
        r'\caption{Final VLM-enabled trials: 10 runs per method and scene, four scenes per domain. Success is planner-reported. Queries, steps and time below are means over successful trials.}',
        r'\label{tab:vlm_final_20261005}',r'\begin{tabular}{llrrrr}',
        r'\hline Domain & Method & Success (\%) & Queries & Steps & Time (s) \\',r'\hline']
    for d in ('tomato','waste'):
        for m in ('pomdp','knowno'):
            r=lookup[d,m,'all']
            tex.append(f"{d.title()} & {'POMDP' if m=='pomdp' else 'KnowNo'} & {r['success_percent']:.2f} & {r['questions_mean_success']:.2f} & {r['steps_mean_success']:.2f} & {r['total_time_s_mean_success']:.2f}"+r' \\')
    tex += [r'\hline',r'\end{tabular}',r'\end{table}']
    (OUT/'paper_table.tex').write_text('\n'.join(tex)+'\n')
    for name in ('plot_comparison.py','plot_all_runs.py'):
        text=(PREVIOUS/name).read_text().replace('vlm_comparison_20261005','vlm_comparison_final_20261005').replace('_vlm_interim_20261005','_vlm_final_20261005')
        text=text.replace('방법별 표본 수·scene 구성 차이를 유지했습니다.','각 방법은 scene 1–4별 10회씩, 총 40회입니다.')
        text=text.replace('방법별 표본 수 및 scene 구성 차이를 유지하여 합산','각 방법은 scene 1–4별 10회씩, 총 40회')
        (OUT/name).write_text(text)
        subprocess.run([sys.executable,str(OUT/name)],check=True)
    integrity=dict(status='passed',selected_runs=160,additional_runs=13,per_cell=10,
        missing_cells=0,duplicate_runs=0,source_hash_mismatches=0,copied_input_files=len(copies))
    (OUT/'integrity_check.json').write_text(json.dumps(integrity,indent=2)+'\n')
    pdf=ROOT/'02_BRL_POMDP_CODE/experiments_logs/analysis_recent/scripts/render_analysis_pdf.py'
    subprocess.run(['python3',str(pdf),str(OUT/'report.md')],check=True)
    (OUT/'README.md').write_text('# VLM 최종 분석 — 2026-10-05\n\n160회 전체가 scene별 10회씩 완료됐다. [report.md](report.md), [report.pdf](report.pdf), [all_scenes_comparison.png](all_scenes_comparison.png)를 참조한다. 성공 실행만의 집계와 성공・실패 전체 집계를 각각 제공한다. `00_raw/`는 분석에 사용한 결과・feedback・planner 로그의 해시 검증 사본이며 영상은 복제하지 않았다. 이전 중간 분석은 보존했다.\n\n재생성:\n```bash\ncd /home/fr/brl\n/home/fr/miniconda3/envs/brl/bin/python collect_exp/vlm_comparison_final_20261005/analyze.py\n```\n')
    mirror=ROOT/'02_BRL_POMDP_CODE/latex_zips/latex_auro/asset/vlm_comparison_final_20261005'
    shutil.copytree(OUT,mirror,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
    (PREVIOUS/'README.md').write_text('# VLM 중간 분석 이력\n\n이 폴더는 마지막 Waste KnowNo 13회가 들어오기 전의 147회 중간 분석이다. 최종 160회 분석은 [../vlm_comparison_final_20261005/report.md](../vlm_comparison_final_20261005/report.md)를 참조한다. 이전 중간 결과는 이력으로 보존했다.\n')
    for t in tests:print(t)
    print(json.dumps(integrity,indent=2))


def report(rows,lookup,tests):
    lines=['# VLM 실험 최종 분석 보고서','', '작성일: 2026-10-05 (Asia/Seoul)', '',
        '## 완료 범위','',
        'Tomato・Waste, POMDP・KnowNo, scene 1–4에 대해 각 10회씩 총 160회이다. 기존 중간 분석 147회에 사용자 지정 Waste KnowNo scene 1의 3회와 scene 2의 10회만 추가했다. 로그의 planner・scenario 및 feedback의 use_vlm=True를 대조했고 중복・누락은 없다.', '',
        '성공은 로그의 Plan Summary.success 판정이다. 영상에 따른 독립 물리 성공 검증은 수행하지 않았다. 지정된 160회를 모두 분모에 포함하며, no user selection으로 끝난 VLM 토큰 파싱 실패 1회도 실패로 포함한다. 기존 제외된 인간 실험, 이전 pilot 및 결과 없는 중단 세션을 새로 포함하지 않았다.', '',
        '## 전체 결과','', '| Domain | Method | 성공/전체 | 성공률 | 95% Wilson CI | 질문(성공 평균 ± SD) | 질의확률(성공, %) | Steps(성공) | 전체 시간(성공, s) |',
        '|---|---|---:|---:|---|---|---|---|---|']
    def avg(r,key,label='success',scale=1):
        mean=r[key+'_mean_'+label];sd=r[key+'_sd_'+label]
        if mean=='':return '—'
        return f'{mean*scale:.2f}' + (f' ± {sd*scale:.2f}' if sd!='' else '')
    for d in ('tomato','waste'):
        for m in ('pomdp','knowno'):
            r=lookup[d,m,'all']
            lines.append(f"| {d} | {m} | {r['successes']}/{r['n']} | {r['success_percent']:.2f}% | {r['ci95_low']:.2f}–{r['ci95_high']:.2f}% | {avg(r,'questions')} | {avg(r,'query_probability',scale=100)} | {avg(r,'steps')} | {avg(r,'total_time_s')} |")
    lines+=['', '## Scene별 성공률','', '| Domain | Scene | POMDP | KnowNo |', '|---|---:|---:|---:|']
    for d in ('tomato','waste'):
        for s in range(1,5):
            a,b=lookup[d,'pomdp',s],lookup[d,'knowno',s]
            lines.append(f"| {d} | {s} | {a['successes']}/10 ({a['success_percent']:.0f}%) | {b['successes']}/10 ({b['success_percent']:.0f}%) |")
    lines+=['', '## 성공률 비교와 해석','', '| Domain | POMDP − KnowNo (%p) | Fisher exact p | Holm p |','|---|---:|---:|---:|']
    for r in tests:lines.append(f"| {r['domain']} | {r['success_delta_pp']:.2f} | {r['fisher_exact_p']:.5g} | {r['holm_p']:.5g} |")
    lines+=['', '동일 seed의 대응 실험으로 확인되지 않아 paired 검정 대신 domain별 비대응 Fisher exact 검정을 사용했다. Holm 보정은 두 domain 비교에 적용했다. 두 방법은 각 scene의 실행 수가 같지만, 성공 조건부 평균의 scene 구성은 성공 수에 따라 달라진다. 실행 간 독립성 가정, 소표본, 방법별 실행 날짜 차이를 고려해 검정은 보조 자료로 읽는다.', '',
        'POMDP는 두 domain에서 각각 37/40(92.5%) 성공했다. KnowNo는 Tomato 10/40(25.0%), Waste 20/40(50.0%)이다. Waste scene 2는 POMDP 10/10, KnowNo 1/10으로 차이가 컸다.', '',
        'KnowNo의 성공 실행 질문 수가 더 적더라도 질문 대상이 다르다. POMDP는 state fact에 대한 Boolean 질문, KnowNo는 action 선택 도움을 요청한다. 질문 한 건의 정보량이 같다고 가정하지 않으며, 낮은 질문 수만으로 효율 우월성을 주장하지 않는다. 실패 실행은 조기 종료할 수 있으므로 성공 평균과 전체 평균을 함께 제공한다.', '',
        '## 시간과 전체 실행 평균','', '| Domain | Method | 질문(전체) | Steps(전체) | 전체 시간(전체, s) | 상호작용(성공, s) | 실행(성공, s) | 탐색(성공, s) | 잔여 시간(성공, s) |','|---|---|---|---|---|---|---|---|']
    for d in ('tomato','waste'):
        for m in ('pomdp','knowno'):
            r=lookup[d,m,'all']
            lines.append(f"| {d} | {m} | {avg(r,'questions','all')} | {avg(r,'steps','all')} | {avg(r,'total_time_s','all')} | {avg(r,'interaction_time_s')} | {avg(r,'execute_time_s')} | {avg(r,'search_time_s')} | {avg(r,'planning_residual_time_s')} |")
    lines+=['', '시간은 플래너 Timing.total_time이며 GUI 세션 전체 시간과 다르다. 상호작용 시간에는 VLM 요청・피드백 처리가 포함된다. 잔여 시간은 시행별 total − interaction − execute이며 search/update/pruning 및 기타 overhead를 포함한다. 순수 탐색 시간으로 치환하지 않는다.', '',
        '## 기록된 실행 설정','',
        '아래는 실행 결과 로그의 metadata에 기록된 값이다. planner의 model과 피드백 VLM의 model을 같은 것으로 추정하지 않는다. 빈 metadata는 미기록으로 표시한다.', '',
        '| Domain | Method | gamma | simulations | planner model | prompt version | generation temperature |',
        '|---|---|---|---|---|---|---|']
    for d in ('tomato','waste'):
        for m in ('pomdp','knowno'):
            group=[r for r in rows if r['domain']==d and r['method']==m]
            params=[', '.join(sorted({r[k] for r in group if r[k]})) or '미기록' for k in ('gamma','n_simulations','model','prompt_version','generation_temperature')]
            lines.append('| '+' | '.join([d,m]+params)+' |')
    lines+=['', '실행별 상세 metadata는 runs.csv에 보존했다. planner metadata가 없는 POMDP 로그는 planner.log의 실행 명령으로 방법을 확인했으며 method_source에 기록했다.', '',
        '## 실패 원인','', '| Domain | Method | 종료 이유 | N |','|---|---|---|---:|']
    counts=Counter((r['domain'],r['method'],r['end_reason']) for r in rows if not r['success'])
    for (d,m,reason),n in sorted(counts.items()):lines.append(f'| {d} | {m} | {reason} | {n} |')
    lines+=['', '추가 Waste KnowNo scene 2 실패 9건 중 8건은 pick에 필요한 detected(w4)가 현재 상태에 없어 종료했다. 다른 1건(20261005_210208_gui)은 VLM이 제공된 후보 토큰 대신 “적절한 행동이 선택지에 없음”이라는 문구를 Token에 반환했고 feedback manager가 후보 토큰 파싱 실패를 기록한 뒤 planner가 no user selection으로 종료했다. 해당 문구만 보고 사람이 선택하지 않은 실패로 분류하지 않는다.', '',
        '이는 로그가 직접 보여주는 실패 지점이다. 영상과 숨겨진 물리 상태를 대조하지 않았으므로 센서 오류・VLM 판단・action 후보 생성 중 어느 하나를 모든 실패의 원인으로 단정하지 않는다.', '',
        '## 추가 13회 실행','', '| Scene | Session | Success | Steps | Questions | End reason |','|---:|---|---|---:|---:|---|']
    for r in rows:
        if r['run'] in LATEST:
            lines.append(f"| {r['scene']} | [{r['run']}]({r['raw_copy']}) | {r['success']} | {r['steps']} | {r['questions']} | {r['end_reason']} |")
    lines+=['', '## 지표・원본・재생성','',
        '- 성공률: 모든 시행 기준. 성공률 CI는 95% Wilson interval.',
        '- 질문・steps・시간: 성공 시행 평균 ± 표본 SD(ddof=1)와 전체 시행 평균을 따로 제공한다. 그림 오차막대는 SD이고 신뢰구간이 아니다.',
        '- 질의확률: 시행별 질문이 발생한 고유 step 수 / 전체 기록 step 수의 평균. 질문 수 / steps와 다르다.',
        '- 상세 scene・성공/전체 지표는 summary.csv, 개별 값은 runs.csv, 실패는 failures.csv이다.',
        '- 00_raw/는 결과 및 feedback/planner 로그의 사본. files_sha256.csv는 원본 경로와 사본 해시를 기록한다.',
        '- 성공 평균 그림은 all_scenes_comparison.png, 전체 시행 그림・표는 all_runs/에 있다.', '',
        '![성공 실행 평균 비교](all_scenes_comparison.png)', '',
        '```bash','cd /home/fr/brl','/home/fr/miniconda3/envs/brl/bin/python collect_exp/vlm_comparison_final_20261005/analyze.py','```']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
