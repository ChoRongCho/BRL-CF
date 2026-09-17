"""Summarize an n_simulations sweep produced by iterate_n_simulations.sh."""
import csv, statistics, sys
from collections import defaultdict
from pathlib import Path

root=Path(sys.argv[1]); runs=list(csv.DictReader((root/'runs.csv').open(newline='',encoding='utf-8'))); groups=defaultdict(list)
for r in runs:
    r.update(total_query='',plan_length='',success='',result_status='execution_error'); logs=list((root/r['run_dir']).glob('exp_*.txt'))
    if r['exit_code']=='0' and len(logs)==1:
        try:
            s=logs[0].read_text(encoding='utf-8').split('[Plan Summary]\n',1)[1].split('[Timing]',1)[0]; f=dict(x.split(': ',1) for x in s.splitlines() if ': ' in x)
            r.update(total_query=int(f['total_questions']),plan_length=int(f['steps']),success=int(f['success']=='True'),result_status='ok')
        except (IndexError,KeyError,ValueError): r['result_status']='invalid_log'
    groups[(r['domain'],r['n_simulations'],'all')].append(r); groups[(r['domain'],r['n_simulations'],r['scene'])].append(r)
rows=[]
for (domain,budget,scene), members in sorted(groups.items()):
    valid=[r for r in members if r['result_status']=='ok']; good=[r for r in valid if r['success']]; row=dict(domain=domain,n_simulations=budget,scene=scene,attempted_runs=len(members),completed_runs=len(valid),error_runs=len(members)-len(valid),success_rate=len(good)/len(valid) if valid else '')
    for metric in ('total_query','plan_length'):
        for suffix, subset in (('',valid),('_success_only',good)):
            values=[r[metric] for r in subset]; row[metric+'_mean'+suffix]=statistics.mean(values) if values else ''; row[metric+'_std'+suffix]=statistics.stdev(values) if len(values)>1 else ''
    rows.append(row)
with (root/'summary.csv').open('w',newline='',encoding='utf-8') as f: w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(f'Saved {root}/summary.csv'); sys.exit(int(any(r['result_status']!='ok' for r in runs)))
