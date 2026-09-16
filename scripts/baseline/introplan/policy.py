"""BRL adaptation of IntroPlan's retrieve -> explain -> score pipeline.

Uses deterministic lexical cosine retrieval instead of the upstream sentence
encoder. The knowledge entries are hand-authored BRL examples, not test labels.
"""
from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
import re


def tokens(text):
    return Counter(re.findall(r"[a-z]+", text.lower()))


def retrieve(prompt, entries, top_k):
    query = tokens(prompt)
    def score(entry):
        vector = tokens(entry['context'])
        denominator = math.sqrt(sum(v*v for v in query.values()) * sum(v*v for v in vector.values()))
        return sum(value * vector[word] for word, value in query.items()) / denominator if denominator else 0
    return sorted(entries, key=score, reverse=True)[:top_k]


class IntrospectiveLLM:
    def __init__(self, call, knowledge_path, domain, trace_path, top_k=3):
        data = json.loads(Path(knowledge_path).read_text())
        self.entries = data[domain]
        if top_k < 1 or not self.entries:
            raise ValueError('A non-empty knowledge base and positive top_k are required')
        for entry in self.entries:
            for key in ('id', 'context', 'options', 'explanation', 'prediction'):
                if not entry.get(key):
                    raise ValueError(f'Knowledge entry is missing {key}')
        self.call = call
        self.top_k = top_k
        self.trace_path = Path(trace_path)

    def __call__(self, prompt, **kwargs):
        if kwargs.get('logprobs') is None:
            return self.call(prompt, **kwargs)
        entries = retrieve(prompt, self.entries, self.top_k)
        examples = '\n\n'.join(
            f"{e['context']}\nOptions:\n{e['options']}\nExplain: {e['explanation']}\nPrediction: {e['prediction']}"
            for e in entries)
        reason_prompt = (
            'Analyze task compliance and uncertainty using only the robot observations. '
            'Unknown properties are not known facts. These are examples from other scenarios:\n\n'
            + examples + '\n\nCurrent decision (ignore its request for a single-letter answer for now):\n'
            + prompt + '\nExplain which options are justified and which need clarification. '
            'Do not invent unobserved facts. Write Explain: followed by your reasoning.'
        )
        reason_response, explanation = self.call(reason_prompt, max_tokens=512, logit_bias={})
        # Upstream CP excludes any direct prediction from the explanation.
        explanation = explanation.split('Prediction:')[0].strip()
        final_prompt = (prompt + '\n\nIntrospective explanation:\n' + explanation
                        + '\nWhich option is correct? Answer with exactly one capital letter A, B, C, D, or E.')
        kwargs = dict(kwargs, logprobs=20, logit_bias={})
        response, text = self.call(final_prompt, **kwargs)
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        with self.trace_path.open('a', encoding='utf-8') as f:
            f.write(json.dumps({'retrieved_ids': [e['id'] for e in entries],
                                'explanation': explanation, 'score_prompt': final_prompt,
                                'reasoning_usage': reason_response.get('usage'),
                                'scoring_usage': response.get('usage')}, ensure_ascii=False) + '\n')
        print('[IntroPlan] Retrieved:', ', '.join(e['id'] for e in entries), flush=True)
        print('[IntroPlan]', explanation, flush=True)
        # Include the additional reasoning call in the shared loop's token totals.
        response = dict(response)
        response['usage'] = {
            key: (response.get('usage') or {}).get(key, 0) + (reason_response.get('usage') or {}).get(key, 0)
            for key in ('prompt_tokens', 'completion_tokens', 'total_tokens')
        }
        return response, text
