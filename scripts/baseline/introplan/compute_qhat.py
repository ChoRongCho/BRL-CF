"""Calibrate IntroPlan on held-out BRL score prompts with labeled options."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import math

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'knowno'))
from scripts.llm import call_llm, configure_openai
from scripts.calibration import score_calibration_choices
# KNOWNO is first on sys.path: reuse its existing CLI calculation helpers.
from compute_qhat import add_scores, load_scored_json, qhat_from_scores, write_csv
from policy import IntrospectiveLLM


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--domain', choices=['tomato', 'wastesorting'], required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--records', help='JSON list of score_prompt and true_options records')
    source.add_argument('--scored-json', help='Previously saved IntroPlan scores; no LLM calls')
    parser.add_argument('--output-csv', default='')
    parser.add_argument('--output', required=True)
    parser.add_argument('--settings', default=str(HERE.parents[2] / 'llm_setting.json'))
    parser.add_argument('--knowledge', default=str(HERE / 'knowledge.json'))
    parser.add_argument('--top-k', type=int, default=3)
    parser.add_argument('--temperature', type=float, default=5.0)
    parser.add_argument('--target-success', type=float, default=0.8)
    parser.add_argument('--quantile-method', choices=['legacy_higher', 'finite_sample'], default='finite_sample')
    args = parser.parse_args()
    records = load_scored_json(args.scored_json or args.records)
    if not records or not 0 < args.target_success < 1 or not math.isfinite(args.temperature) or args.temperature <= 0:
        parser.error('Need nonempty records, positive temperature, and 0 < target-success < 1')
    if args.top_k < 1:
        parser.error('top-k must be positive')
    for record in records:
        if not record.get('true_options') or not set(record['true_options']) <= set('ABCDE'):
            parser.error('Each record needs true_options (A-E)')
        if not args.scored_json:
            prompt = record.get('score_prompt') or record.get('mc_score_prompt')
            if not prompt:
                parser.error('Each unscored record needs score_prompt or mc_score_prompt')
            record['mc_score_prompt'] = prompt
    if not args.scored_json:
        configure_openai(settings_path=args.settings)
        policy = IntrospectiveLLM(call_llm, args.knowledge, args.domain, args.output + '.trace.jsonl', args.top_k)
        score_calibration_choices(records, logprobs_count=20, llm_call=policy, logit_bias={})
    rows = add_scores(records, args.temperature)
    qhat, q_level = qhat_from_scores(records, args.target_success, method=args.quantile_method)
    write_csv(args.output_csv, rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(qhat=qhat, n=len(records), domain=args.domain,
                                     temperature=args.temperature, target_success=args.target_success,
                                     knowledge=args.knowledge, top_k=args.top_k, records=records,
                                     q_level=q_level, quantile_method=args.quantile_method, baseline='introplan'), indent=2)+'\n')
    print('qhat:', qhat, 'Saved:', output)


if __name__ == '__main__':
    main()
