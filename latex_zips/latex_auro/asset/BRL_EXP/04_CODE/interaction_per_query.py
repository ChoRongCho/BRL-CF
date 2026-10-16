"""Pooled interaction time per query across all successful and failed trials."""
from collections import defaultdict
from build_data import read, write

METRIC = 'interaction_time_per_query_all_s'
FILENAME = 'exp_human_interaction_time_per_query_all.pdf'


def build_statistics(package):
    rows = read(package/'01_BASELINE/01_processed/human_summary.csv')
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['domain'],row['condition']].append(row)
    result = []
    for domain, label in [('waste','WasteSorting'),('tomato','TomatoHarvesting')]:
        for condition, method in [('knowno','KnowNo'),('pomdp','Ours')]:
            group = grouped[domain,condition]
            queried = [r for r in group if int(r['query']) > 0]
            query_count = sum(int(r['query']) for r in queried)
            interaction_total = sum(float(r['interaction_time']) for r in queried)
            if not query_count:
                raise ValueError(f'No trials with queries: {domain}/{condition}')
            result.append(dict(environment='Human',domain=label,method=method,mode='all',metric=METRIC,
                               total_n=len(group),success_n=sum(r['success'].upper()=='TRUE' for r in group),
                               metric_n=query_count,trial_n=len(group),interaction_time_total_s=interaction_total,excluded_zero_query_n=sum(float(r['query'])==0 for r in group),
                               mean=interaction_total/query_count,sample_sd='',
                               aggregation='total_interaction_time / total_queries',error_type='not_reported',
                               ci95_low='',ci95_high='',unit='seconds / query'))
    write(package/'01_BASELINE/02_graph_data/human_interaction_per_query_all.csv', result)
    return result
