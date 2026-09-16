"""IntroPlan calibration on exactly the recovered historical KnowNo candidates."""
from pathlib import Path
import argparse, concurrent.futures, hashlib, json, sys
ROOT=Path(__file__).resolve().parents[4]
BASE=ROOT/'scripts/baseline'
sys.path[:0]=[str(BASE/'knowno'),str(BASE/'introplan')]
from compute_qhat import add_scores, qhat_from_scores
from scripts.calibration import score_calibration_choices
from scripts.llm import configure_openai, call_llm
from policy import IntrospectiveLLM
OUT=Path(__file__).resolve().parent

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def atomic_write(p,data):
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(p)
def score_one(payload):
 domain,index,record=payload
 configure_openai(settings_path=ROOT/'llm_setting.json')
 scorer=IntrospectiveLLM(call_llm,OUT/'knowledge_snapshot.json',domain,OUT/f'{domain}_traces/{index:03d}.jsonl',3)
 score_calibration_choices([record],20,llm_call=scorer,logit_bias={})
 return index,record

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--domain',choices=['tomato','wastesorting'],required=True)
 args=parser.parse_args();domain=args.domain
 inputs=json.loads((OUT/f'{domain}_inputs.json').read_text())
 fingerprint=json.loads((OUT/f'{domain}_manifest.json').read_text())
 current=configure_openai(settings_path=ROOT/'llm_setting.json')
 assert current.get('model')==fingerprint['model']
 assert sha(OUT/'knowledge_snapshot.json')==fingerprint['knowledge_sha256']
 assert sha(OUT/f'{domain}_inputs.json')==fingerprint['inputs_sha256']
 checkpoint=OUT/f'{domain}_checkpoint.json'
 done=json.loads(checkpoint.read_text()) if checkpoint.exists() else {}
 todo=[(domain,index,record) for index,record in enumerate(inputs) if str(index) not in done]
 # Separate processes keep the existing signal-based LLM timeout valid.
 with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:
  futures=[pool.submit(score_one,payload) for payload in todo]
  for future in concurrent.futures.as_completed(futures):
   index,record=future.result();done[str(index)]=record;atomic_write(checkpoint,done)
   print(f'PROGRESS {domain} {len(done)}/100',flush=True)
 records=[done[str(i)] for i in range(100)]
 add_scores(records,5.0)
 targets={}
 for method in ['legacy_higher','finite_sample']:
  targets[method]={}
  for target in [.75,.8,.85,.95]:
   qhat,level=qhat_from_scores(records,target,method)
   targets[method][str(target)]={'qhat':qhat,'q_level':level,'calibration_coverage':sum(r['qhat_score']<=qhat for r in records)/100}
 result=dict(fingerprint,records=records,targets=targets,n=100,baseline='introplan',
             qhat=targets['legacy_higher']['0.95']['qhat'],quantile_method='legacy_higher',target_success=.95)
 atomic_write(OUT/f'{domain}_result.json',result)
 print('RESULT',domain,json.dumps(targets),flush=True)
if __name__=='__main__':main()
