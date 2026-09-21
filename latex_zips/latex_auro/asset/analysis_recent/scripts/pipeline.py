"""Frozen raw-source manifest -> episodes -> figure data -> figures.
Run from any directory: python pipeline.py --stage all
"""
from pathlib import Path
import argparse, collections, csv, hashlib, importlib.util, json, math, os, re, statistics
BASE = Path(__file__).resolve().parents[1]
PROJECT = BASE.parents[1]
LOGS = PROJECT/'experiments_logs'
SPEC = importlib.util.spec_from_file_location('raw_parser', Path(__file__).resolve().parent/'raw_parser_snapshot.py')
parser = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(parser)
METRICS = {
 'success': ('success', 'Task success (%)', False),
 'questions': ('questions', 'Questions per episode', False),
 'steps': ('steps', 'Physical/planning steps', False),
 'seconds': ('seconds', 'Logged episode time (s)', False),
 'questions_success_only': ('questions', 'Questions (successful episodes)', True),
 'steps_success_only': ('steps', 'Steps (successful episodes)', True),
 'reward': ('reward', 'Cumulative reward', False),
}

def write(path, rows):
 path.parent.mkdir(parents=True, exist_ok=True)
 fields=list(dict.fromkeys(k for r in rows for k in r))
 with path.open('w',newline='') as f:
  w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)

def read(path):
 with path.open(newline='') as f:return list(csv.DictReader(f))

def stage1(package):
 config=json.loads((package/'manifest.json').read_text()); rows=[]
 for i, item in enumerate(config['sources'],1):
  p=PROJECT/item['source_file']
  digest=hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else ''
  if digest!=item['sha256']: raise ValueError('Raw source changed: '+str(p))
  text=p.read_text()
  block=re.search(r'\[Meta\]\s*\n(.*?)(?=\n\[|\Z)', text, re.S)
  meta=dict(line.split(': ',1) for line in block[1].splitlines() if ': ' in line) if block else {}
  row=dict(index=i,domain=item['domain'],condition=item['condition'],scene=item['scene'],
           iteration=item.get('iteration',''),seed=item.get('seed',''),raw_source=item['source_file'],
           raw_sha256=digest,status='invalid_log',success='',questions='',steps='',seconds='',reward='',
           end_reason='',gamma=meta.get('gamma',''),n_simulations=meta.get('n_simulations',''),
           threshold=meta.get('threshold',''),timestamp=parser.timestamp_from_name(p))
  if item.get('exit_code','0')!='0':row['status']='execution_error'
  elif item.get('parser')=='baseline':
   d=parser.parse_knowno_log(p,item['domain'],i,package/'00_raw'/item['domain'])
   if d:
    row.update(status='ok',success=int(d['success']),questions=d['question_count'],steps=d['planning_length'],
               seconds=d['elapsed_seconds'],seed=d['seed'],end_reason=d['stop_reason'],model=d['model'],prompt_version=d['prompt_version'])
  else:
   d=parser.parse_section_key_values(text,'Plan Summary');t=parser.parse_timing_values(text)
   if all(k in d for k in ['success','steps','total_questions']):
    row.update(status='ok',success=int(d['success']),questions=int(d['total_questions']),steps=int(d['steps']),
      seconds=t.get('total_time',''),reward=d.get('cumulated_reward',''),end_reason=d.get('end_reason',''),seed=meta.get('seed',row['seed']))
  rows.append(row)
 write(package/'01_processed/episodes.csv',rows)
 write(package/'00_raw_sources.csv',[dict(source_file=r['raw_source'],sha256=r['raw_sha256'],domain=r['domain'],condition=r['condition'],scene=r['scene']) for r in rows])
 # One raw log per slot; do not silently count backups/retries as new episodes.
 keys=[(r['domain'],r['condition'],r['scene'],str(r['seed'])) for r in rows if r['seed']!='']
 if len(keys)!=len(set(keys)):raise ValueError('Duplicate scene/seed/condition: '+package.name)
 print(package.name, 'episodes',len(rows), 'valid',sum(r['status']=='ok' for r in rows),flush=True)

def stage2(package):
 config=json.loads((package/'manifest.json').read_text()); episodes=read(package/'01_processed/episodes.csv'); output=[]
 for domain in ['tomato','wastesorting']:
  for order,condition in enumerate(config['conditions']):
   attempted=[r for r in episodes if r['domain']==domain and r['condition']==condition]
   for metric,(field,label,success_only) in METRICS.items():
    valid=[r for r in attempted if r['status']=='ok' and (not success_only or r['success']=='1') and r[field]!='']
    values=[float(r[field]) for r in valid];n=len(values)
    value=lower=upper=minus=plus=sd=''; error='95% Wilson CI' if metric=='success' else 'mean +/- 1 SE'
    if n:
     value=statistics.mean(values);sd=statistics.stdev(values) if n>1 else ''
     if metric=='success':
      z=1.959963984540054;den=1+z*z/n
      center=(value+z*z/(2*n))/den;half=z*math.sqrt(value*(1-value)/n+z*z/(4*n*n))/den
      lower=max(0,100*(center-half));upper=min(100,100*(center+half));value*=100
     elif n>1:lower=value-sd/math.sqrt(n);upper=value+sd/math.sqrt(n)
     if lower!='':minus=max(0,value-lower);plus=max(0,upper-value)
    output.append(dict(domain=domain,condition=condition,order=order,metric=metric,metric_label=label,
      n_attempted=len(attempted),n_valid=n,n_execution_errors=sum(r['status']!='ok' for r in attempted),
      n_missing_metric=sum(r['status']=='ok' and (not success_only or r['success']=='1') and r[field]=='' for r in attempted),
      success_only=int(success_only),numerator=sum(values) if values else '',value=value,sd=sd,lower=lower,upper=upper,
      error_minus=minus,error_plus=plus,error_type=error,source_csv='01_processed/episodes.csv'))
 write(package/'02_graph_data/figure_data.csv',output)
 from detailed_analysis import analyze_details
 analyze_details(package, config, episodes)


def stage3(package):
 os.environ.setdefault('MPLCONFIGDIR','/tmp/brl-matplotlib')
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
 config=json.loads((package/'manifest.json').read_text());data=read(package/'02_graph_data/figure_data.csv')
 conditions=config['conditions'];sweep=config['sweep'];colors=['#197C80','#D9993C']
 out=package/'03_figures';out.mkdir(exist_ok=True)
 def draw(ax,metric):
  for j,domain in enumerate(['tomato','wastesorting']):
   rows=sorted([r for r in data if r['domain']==domain and r['metric']==metric and r['value']!=''],key=lambda r:int(r['order']))
   if not rows:continue
   x=[float(r['condition']) if sweep else int(r['order'])+(-.19 if j==0 else .19) for r in rows]
   y=[float(r['value']) for r in rows]
   errors=[[float(r[k]) if r[k]!='' else 0 for r in rows] for k in ['error_minus','error_plus']]
   label='Tomato' if j==0 else 'Waste'
   if sweep:ax.errorbar(x,y,yerr=errors,label=label,color=colors[j],marker='o',markersize=4,capsize=3,lw=1.4)
   else:ax.bar(x,y,width=.36,yerr=errors,label=label,color=colors[j],capsize=3)
  ax.set_ylabel(METRICS[metric][1]);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
  if sweep:ax.set_xlabel(config['axis']);ax.set_xticks([float(c) for c in conditions]);ax.tick_params(axis='x',labelrotation=30 if len(conditions)>7 else 0)
  else:ax.set_xticks(range(len(conditions)),[config.get('labels',{}).get(c,c) for c in conditions],rotation=20,ha='right')
  if metric=='success':ax.set_ylim(0,110)
  elif metric!='reward':ax.set_ylim(bottom=0)
  ax.legend(frameon=False,fontsize=9)
 def save(fig,name):
  fig.text(.08,.025,'Success: 95% Wilson CI; other metrics: mean ± SE. Failed tasks included unless marked success-only.',fontsize=8)
  for ext in ['png','pdf']:fig.savefig(out/(name+'.'+ext),dpi=200,facecolor='white')
  plt.close(fig)
 fig,axes=plt.subplots(2,2,figsize=(12,8.4))
 for ax,metric in zip(axes.flat,['success','questions','steps','seconds']):draw(ax,metric)
 fig.suptitle(config['title'],fontsize=16);fig.tight_layout(rect=[0,.06,1,.95]);save(fig,'overview')
 for metric in METRICS:
  if not any(r['metric']==metric and r['value']!='' for r in data):continue
  fig,ax=plt.subplots(figsize=(8,5));draw(ax,metric);ax.set_title(config['title']);fig.tight_layout(rect=[0,.07,1,1]);save(fig,metric)


def docs(package):
 c=json.loads((package/'manifest.json').read_text());rows=read(package/'01_processed/episodes.csv')
 dates=sorted({r['timestamp'][:10] for r in rows if r['timestamp']})
 params={k:sorted({r.get(k,'') for r in rows if r.get(k,'')!=''}) for k in ['gamma','n_simulations','threshold']}
 groups=collections.Counter((r['domain'],r['condition'],r['status']) for r in rows)
 lines=['# '+c['title'],'',c['notes'],'',f'실행 로그 {len(rows)}건. 파일명에서 확인된 실행일: {", ".join(dates) or "미상"}.','',
 '## 파일 구조','','- Raw: `00_raw/`에 실제 원본 파일을 보관(핵심 4개 실험). `00_raw_sources.csv`는 프로젝트 루트 기준 경로와 해시. 원본 내용은 변경하지 않음.',
 '- `manifest.json`: 선택한 raw 파일, 조건, 배치 경계와 해시.',
 '- `01_processed/episodes.csv`: 실행별 수치와 raw_source/raw_sha256. 오류/결측은 0으로 바꾸지 않음.',
 '- `01_processed/summary.csv`, `scenes.csv`, `paired.csv`, `paired_episodes.csv`: 전체·장면별 집계 및 동일 scene/seed 비교. 상세 해석은 `report.md`.',
 '- `02_graph_data/figure_data.csv`: 그림의 각 점/막대, 평균/비율, 표본 수, 오차막대 수치. 그래프는 이 CSV를 직접 읽음.',
 '- `03_figures/overview.png`, `.pdf`: 성공률·질문·행동·시간 비교. 개별 지표 그림도 제공.',
 '', '## 집계 기준','','정상 종료한 과제 실패도 평균에 포함. success-only 지표만 성공 실행으로 제한. 성공률은 유효 결과 기준으로 계산하며 오류 건수는 별도 표기. 시간은 각 로그의 시간 정의를 따르며 실제 사람 응답 시간으로 해석하지 않음. 기존 논문 그림의 오차막대/필터와 같다고 가정하지 말 것.',
 '', '원본 파라미터: `'+json.dumps(params,ensure_ascii=False)+'`','',
 '| Domain | Condition | Status | n |','|---|---|---|---:|']
 lines += ['| '+' | '.join([*k,str(n)])+' |' for k,n in sorted(groups.items())]
 lines += ['', '## 재생성','','프로젝트 루트에서:', '', '```bash',f'python3 experiments_logs/analysis_recent/scripts/pipeline.py --package {package.name} --stage all','```','',
 '단계별로 `--stage 1`, `--stage 2`, `--stage 3` 실행 가능. 원본 해시가 바뀌면 자동 중단. 오래된 배치와 최신 배치는 합치지 않음.']
 (package/'README.md').write_text('\n'.join(lines)+'\n')

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['all','1','2','3'],default='all');ap.add_argument('--package',default='all');args=ap.parse_args()
 for path in sorted(BASE.glob('*/manifest.json')):
  package=path.parent
  if args.package!='all' and package.name!=args.package:continue
  for n,fn in [('1',stage1),('2',stage2),('3',stage3)]:
   if args.stage in ['all',n]:fn(package)
  docs(package)
if __name__=='__main__':main()
