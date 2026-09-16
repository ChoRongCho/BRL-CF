"""Reproducible calibration of both BRL scoring policies on fixed candidates."""
from pathlib import Path
import copy, hashlib, json, sys
ROOT=Path(__file__).resolve().parents[4]
BASE=ROOT/'scripts/baseline'
sys.path[:0]=[str(BASE/'knowno'), str(BASE/'introplan')]
from compute_qhat import DOMAIN_CONFIG, add_scores, qhat_from_scores
from scripts.calibration import _load_text_records, prepare_calibration_choices, score_calibration_choices
from scripts.llm import configure_openai, call_llm
from policy import IntrospectiveLLM
OUT=Path(__file__).resolve().parent

def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
 settings=configure_openai(settings_path=ROOT/'llm_setting.json')
 results=[]
 for domain,filename in [('tomato','tomato'),('wastesorting','waste')]:
  # Keep this historical run tied to its original 18/23-record input.
  source=BASE/'calibration/recovery_20260915'/f'{filename}-mc-gen-prompt.txt'
  original=_load_text_records(source.read_text())
  config=DOMAIN_CONFIG[domain]
  prepare_calibration_choices(original,config['background'],config['builder'],generate=False)
  for r in original:
   assert len(set(r['mc_gen_all']))==len(r['mc_gen_all']), 'duplicate candidates'
  (OUT/f'{domain}_prompts.json').write_text(json.dumps(original,indent=2))
  for baseline in ['knowno','introplan']:
   checkpoint=OUT/f'{baseline}_{domain}_scored.json'
   records=json.loads(checkpoint.read_text()) if checkpoint.exists() else []
   scorer=call_llm if baseline=='knowno' else IntrospectiveLLM(call_llm, BASE/'introplan/knowledge.json',domain,OUT/f'{baseline}_{domain}_trace.jsonl')
   for index in range(len(records),len(original)):
    r=copy.deepcopy(original[index])
    score_calibration_choices([r],logprobs_count=5 if baseline=='knowno' else 20,llm_call=scorer,logit_bias={})
    records.append(r)
    checkpoint.write_text(json.dumps(records,indent=2))
    print(f'PROGRESS {baseline} {domain} {index+1}/{len(original)}',flush=True)
   add_scores(records,5.0)
   targets={}
   for target in [.8,.85,.95]:
    qhat,q_level=qhat_from_scores(records,target,method='finite_sample')
    covered=sum(r['qhat_score']<=qhat for r in records)/len(records)
    targets[str(target)]={'qhat':qhat,'q_level':q_level,'calibration_coverage':covered}
   result={'baseline':baseline,'domain':domain,'model':settings.get('model'),'temperature':5.0,
           'n':len(records),'unique_contexts':len(set(r['context'] for r in original)),
           'quantile_method':'finite_sample','targets':targets,'records':records,
           'source':str(source.relative_to(ROOT)),'source_sha256':digest(source),
           'knowledge_sha256':digest(BASE/'introplan/knowledge.json') if baseline=='introplan' else None,
           'top_k':3 if baseline=='introplan' else None,'score_top_logprobs':5 if baseline=='knowno' else 20,
           'scope':'Existing fixed-candidate calibration prompts; not independent rollout evaluation.'}
   (OUT/f'{baseline}_{domain}_result.json').write_text(json.dumps(result,indent=2))
   results.append({k:v for k,v in result.items() if k!='records'})
   (OUT/'results.json').write_text(json.dumps(results,indent=2))
   print('RESULT',baseline,domain,targets,flush=True)
if __name__=='__main__': main()
