"""Verify moved raw bytes and all four paper data products."""
from pathlib import Path
import csv,hashlib,json,math,statistics
BASE=Path(__file__).resolve().parents[1];PROJECT=BASE.parents[1]
NAMES=['threshold_20260912','when_what_original_20260912','when_what_mechanisms_20260921','query_baselines_current_20260920']
def read(p):return list(csv.DictReader(p.open()))
move=json.loads((BASE/'raw_migration.json').read_text())
assert move['status']=='complete'
for r in move['files']:
 f=PROJECT/r['new']
 assert f.is_file() and not f.is_symlink(),f
 assert hashlib.sha256(f.read_bytes()).hexdigest()==r['sha256'],f
 assert not (PROJECT/r['old']).exists(),r['old']
results=[]
for name in NAMES:
 d=BASE/name;assert not any(p.is_symlink() for p in d.rglob('*')),name
 required=['episodes.csv','summary.csv','scenes.csv','paired.csv','paired_episodes.csv']
 assert all((d/'01_processed'/f).is_file() for f in required)
 episodes=read(d/'01_processed/episodes.csv');g=read(d/'02_graph_data/figure_data.csv');summary=read(d/'01_processed/summary.csv');pairs=read(d/'01_processed/paired.csv')
 for r in episodes:
  f=PROJECT/r['raw_source'];f.relative_to(d/'00_raw')
  assert hashlib.sha256(f.read_bytes()).hexdigest()==r['raw_sha256']
 for r in g:
  field=r['metric'].replace('_success_only','')
  rs=[x for x in episodes if x['domain']==r['domain'] and x['condition']==r['condition'] and x['status']=='ok' and (r['success_only']=='0' or x['success']=='1') and x[field]!='']
  assert len(rs)==int(r['n_valid'])
  if rs:
   values=[float(x[field]) for x in rs];mean=statistics.mean(values)*(100 if field=='success' else 1)
   assert math.isclose(mean,float(r['value']),abs_tol=1e-10)
   if r['lower']!='':assert float(r['lower'])<=mean+1e-10 and float(r['upper'])>=mean-1e-10
  else:assert r['value']==''
 for r in summary:
  rs=[x for x in episodes if x['condition']==r['condition'] and (r['domain']=='all' or x['domain']==r['domain']) and x['status']=='ok']
  assert len(rs)==int(r['n_valid']) and sum(int(x['success']) for x in rs)==int(r['successes'])
 for r in pairs:
  maps=[]
  for c in [r['reference'],r['comparison']]:maps.append({(x['domain'],x['scene'],x['seed']):x for x in episodes if x['condition']==c and x['status']=='ok' and (r['domain']=='all' or r['domain']==x['domain'])})
  a,b=maps;keys=a.keys()&b.keys();assert len(keys)==int(r['n_pairs'])
  assert sum(int(a[k]['success'])>int(b[k]['success']) for k in keys)==int(r['reference_only_success'])
  assert sum(int(a[k]['success'])<int(b[k]['success']) for k in keys)==int(r['comparison_only_success'])
 for ext in ['png','pdf']:assert (d/'03_figures'/('overview.'+ext)).stat().st_size>1000
 assert (d/'report.md').stat().st_size>1000
 results.append(dict(experiment=name,episodes=len(episodes),valid=sum(r['status']=='ok' for r in episodes),raw_links=0,graph_rows=len(g),paired_comparisons=len(pairs)))
(BASE/'raw_integrity_check.json').write_text(json.dumps(dict(moved_files=len(move['files']),missing=0,hash_mismatch=0,results=results),indent=2)+'\n')
print(json.dumps(results,indent=2));print('All moved raw hashes, CSV aggregates, paired matches and figures verified.')
