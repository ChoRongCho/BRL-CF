"""Common group/scene and paired-seed analyses for all experiment packages."""
from collections import Counter
from pathlib import Path
import csv, math, statistics as st

FIELDS=('questions','steps','seconds','reward')

def write(path,rows):
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with path.open('w',newline='') as f:
  w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)

def wilson(success,n):
 if not n:return '', ''
 p=success/n;z=1.959963984540054;den=1+z*z/n
 center=(p+z*z/(2*n))/den;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
 return max(0,100*(center-half)),min(100,100*(center+half))

def summarize(rows,domain,condition,scene='all'):
 valid=[r for r in rows if r['status']=='ok'];success=sum(int(r['success']) for r in valid)
 lo,hi=wilson(success,len(valid))
 out=dict(domain=domain,condition=condition,scene=scene,n_attempted=len(rows),n_valid=len(valid),n_errors=len(rows)-len(valid),successes=success,
  success_percent=100*success/len(valid) if valid else '',success_ci_low=lo,success_ci_high=hi,
  success_percent_errors_as_failure=100*success/len(rows) if rows else '',end_reasons=str(dict(Counter(r['end_reason'] for r in valid))))
 for field in FIELDS:
  for suffix,subset in [('',valid),('_success_only',[r for r in valid if r['success']=='1'])]:
   values=[float(r[field]) for r in subset if r[field]!=''];n=len(values)
   out[field+suffix+'_n']=n;out[field+suffix+'_mean']=st.mean(values) if n else ''
   out[field+suffix+'_sd']=st.stdev(values) if n>1 else ''
 return out

def fmt(v):return '—' if v=='' else f'{v:.3g}' if isinstance(v,float) else str(v)

def analyze_details(package,config,episodes):
 conditions=config['conditions'];summary=[];scenes=[]
 for domain in ['all','tomato','wastesorting']:
  for condition in conditions:
   subset=[r for r in episodes if r['condition']==condition and (domain=='all' or r['domain']==domain)]
   summary.append(summarize(subset,domain,condition))
   if domain!='all':
    for scene in sorted({r['scene'] for r in subset}):scenes.append(summarize([r for r in subset if r['scene']==scene],domain,condition,scene))
 write(package/'01_processed/summary.csv',summary);write(package/'01_processed/scenes.csv',scenes)
 reference='0.8' if package.name.startswith('threshold_') else 'ours' if 'ours' in conditions else conditions[0]
 pairs=[];pair_details=[]
 for domain in ['all','tomato','wastesorting']:
  lookup={}
  for r in episodes:
   if r['status']=='ok' and r['seed']!='' and (domain=='all' or r['domain']==domain):
    lookup.setdefault(r['condition'],{})[(r['domain'],r['scene'],r['seed'])]=r
  a=lookup.get(reference,{})
  for condition in conditions:
   if condition==reference:continue
   b=lookup.get(condition,{});keys=sorted(a.keys() & b.keys())
   wins=sum(int(a[k]['success'])>int(b[k]['success']) for k in keys);losses=sum(int(a[k]['success'])<int(b[k]['success']) for k in keys);discord=wins+losses
   p=min(1.,2*sum(math.comb(discord,j) for j in range(min(wins,losses)+1))/2**discord) if discord else 1. if keys else ''
   row=dict(domain=domain,reference=reference,comparison=condition,n_pairs=len(keys),reference_unmatched=len(a)-len(keys),comparison_unmatched=len(b)-len(keys),reference_only_success=wins,comparison_only_success=losses,success_delta_pp=100*(wins-losses)/len(keys) if keys else '',exact_mcnemar_p=p,p_holm='',difference_direction='reference minus comparison')
   for field in FIELDS:
    ds=[float(a[k][field])-float(b[k][field]) for k in keys if a[k][field]!='' and b[k][field]!=''];n=len(ds)
    mean=st.mean(ds) if n else '';half=1.959963984540054*st.stdev(ds)/math.sqrt(n) if n>1 else ''
    row[field+'_paired_n']=n;row[field+'_delta_mean']=mean
    row[field+'_delta_ci95_normal_low']=mean-half if half!='' else '';row[field+'_delta_ci95_normal_high']=mean+half if half!='' else ''
   pairs.append(row)
   if domain!='all':
    for key in keys:
     pair_details.append(dict(domain=domain,scene=key[1],seed=key[2],reference=reference,comparison=condition,reference_source=a[key]['raw_source'],comparison_source=b[key]['raw_source'],reference_success=a[key]['success'],comparison_success=b[key]['success'],**{field+'_delta':float(a[key][field])-float(b[key][field]) if a[key][field]!='' and b[key][field]!='' else '' for field in FIELDS}))
  selected=sorted([r for r in pairs if r['domain']==domain and r['exact_mcnemar_p']!=''],key=lambda r:r['exact_mcnemar_p']);prev=0.
  for i,r in enumerate(selected):prev=max(prev,min(1.,(len(selected)-i)*r['exact_mcnemar_p']));r['p_holm']=prev
 write(package/'01_processed/paired.csv',pairs);write(package/'01_processed/paired_episodes.csv',pair_details)
 lines=['# '+config['title']+' — 분석 보고서','',config['notes'],'',
 f'실행 슬롯 {len(episodes):,}개, 유효 결과 {sum(r["status"]=="ok" for r in episodes):,}개. Raw는 `00_raw/`, 각 수치의 출처는 episodes.csv의 raw_source이다.','',
 '## 전체·도메인별 결과','', '| Domain | Condition | 성공/유효 | 성공률 % | 질문 평균 | 행동 평균 | 시간 평균(s) | 오류 |', '|---|---|---:|---:|---:|---:|---:|---:|']
 for r in summary:lines.append('| '+' | '.join([r['domain'],r['condition'],f"{r['successes']}/{r['n_valid']}",fmt(r['success_percent']),fmt(r['questions_mean']),fmt(r['steps_mean']),fmt(r['seconds_mean']),str(r['n_errors'])])+' |')
 lines+=['','## 장면별 결과','','| Domain | Scene | Condition | 성공/유효 | 질문 평균 | 행동 평균 |','|---|---|---|---:|---:|---:|']
 for r in scenes:lines.append('| '+' | '.join([r['domain'],r['scene'],r['condition'],f"{r['successes']}/{r['n_valid']}",fmt(r['questions_mean']),fmt(r['steps_mean'])])+' |')
 lines+=['','## 동일 scene·seed paired 비교','',f'기준 조건: `{reference}`. 모든 차이는 기준 − 비교 조건. 양쪽 결과가 유효하고 domain/scene/seed가 일치하는 쌍만 사용한다.','',
 '| Domain | Comparison | 쌍 수 | 기준만 성공 | 상대만 성공 | 성공률 차이(pp) | McNemar p | Holm p | 질문 차이 |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
 for r in pairs:lines.append('| '+' | '.join([r['domain'],r['comparison'],str(r['n_pairs']),str(r['reference_only_success']),str(r['comparison_only_success']),fmt(r['success_delta_pp']),fmt(r['exact_mcnemar_p']),fmt(r['p_holm']),fmt(r['questions_delta_mean'])])+' |')
 lines+=['','## 해석 및 제한','',
 '- 성공률 p는 양측 exact McNemar. Holm 보정은 이 보고서의 각 domain 내 기준 조건 대비 비교군에 적용. all/domain 검정을 하나의 독립 증거로 중복 해석하지 않는다.',
 '- 질문·행동·시간 차이의 CI는 paired 차이 평균의 정규근사 95% 구간. 그래프의 개별 평균 오차막대는 ±1 SE이며 서로 다른 통계이다.',
 '- 과제 실패도 전체 평균에 포함한다. 적은 행동/질문은 조기 실패의 결과일 수 있으므로 성공률과 함께 해석한다. 성공 실행만의 평균은 summary/scenes CSV의 *_success_only 열에 분리했다.',
 '- seed를 맞춰도 정책 경로가 달라진 이후 같은 난수 사건까지 보장하지는 않는다. LLM 비결정성과 서로 다른 실행 날짜·파라미터도 고려해야 한다.',
 '- 그림은 02_graph_data/figure_data.csv의 값과 오차막대를 직접 읽는다. 원본 오류/결측을 0으로 대체하지 않는다.']
 overall=[r for r in summary if r['domain']=='all' and r['n_valid']]
 if overall:
  best=max(overall,key=lambda r:r['success_percent']);fewest=min(overall,key=lambda r:r['questions_mean'])
  lines += ['',f'관측 결과: 전체 최고 성공률은 {best["condition"]} ({best["success_percent"]:.2f}%), 최소 평균 질문은 {fewest["condition"]} ({fewest["questions_mean"]:.3f}회). 이 순위만으로 통계적 우월성이나 인과를 주장하지 않는다.']
 # Show logged parameters per condition instead of concealing differences between reruns.
 lines+=['','## 기록된 설정','','| Condition | gamma | simulations | threshold | 모델 |','|---|---|---|---|---|']
 for c in conditions:
  subset=[r for r in episodes if r['condition']==c]
  vals=[', '.join(sorted({r.get(k,'') for r in subset if r.get(k,'')!=''})) or '미기록' for k in ['gamma','n_simulations','threshold','model']]
  lines.append('| '+' | '.join([c]+vals)+' |')
 errors=[r for r in episodes if r['status']!='ok']
 if errors:
  lines+=['', '## 실행 오류 원본', '']
  for r in errors:lines.append('- '+r['domain']+' / '+r['condition']+' / scene '+r['scene']+' / seed '+r['seed']+': `'+r['raw_source']+'`')
 (package/'report.md').write_text('\n'.join(lines)+'\n')
