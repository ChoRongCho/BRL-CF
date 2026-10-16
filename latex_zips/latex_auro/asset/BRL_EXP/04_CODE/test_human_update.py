"""Check 24 -> 32 -> 40 trial growth with cloned logs in a temporary fixture.

Synthetic p004/p005 fixtures never enter the real research dataset.
"""
from pathlib import Path
import hashlib
import json
import shutil
import tempfile

import aggregate_human_summary as aggregate
import build_data as build
import summarize_human as summary
import update_human as update


def main():
    real = update.ROOT
    original_files = {p: hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in (real.parents[1]/'figures').rglob('exp_*.pdf')}
    with tempfile.TemporaryDirectory(prefix='brl-human-update-') as directory:
        temp = Path(directory)
        source = temp/'human'
        package = temp/'latex_auro/asset/BRL_EXP'
        (package/'01_BASELINE/00_raw').mkdir(parents=True)
        (package/'source_files.csv').write_bytes((real/'source_files.csv').read_bytes())
        for participant in ['p001','p002','p003']:
            shutil.copytree(real/'01_BASELINE/00_raw/human'/participant, source/participant)
        actual_root = build.ROOT
        try:
            build.ROOT = package
            for participant, expected in [(None,24),('p004',32),('p005',40)]:
                if participant:
                    shutil.copytree(source/'p001', source/participant)
                rows = summary.collect_rows(source)
                build.write(source/'human_summary.csv', [{k:summary.display(r[k]) for k in summary.COLUMNS} for r in rows])
                build.write(source/'human_aggregate.csv', [{k:aggregate.display(r[k]) for k in aggregate.COLUMNS}
                            for r in aggregate.aggregate(aggregate.read_rows(source/'human_summary.csv'))])
                assert update.sync_human(source, package)==expected
                normalized = build.human_runs()
                assert len(normalized)==expected
                stats = build.stats(normalized,['environment','domain','method'])
                assert {r['total_n'] for r in stats}=={expected//4}
            # Supply unchanged Oracle/VLM graph inputs, plus synthetic Human stats.
            graph = package/'01_BASELINE/02_graph_data'
            vlm = [r for r in build.read(real/'01_BASELINE/02_graph_data/vlm_human_statistics.csv') if r['environment']=='VLM']
            oracle = [r for r in build.read(real/'01_BASELINE/02_graph_data/response_comparison_statistics.csv') if r['environment']=='Oracle']
            build.write(graph/'vlm_human_statistics.csv', vlm+stats)
            build.write(graph/'response_comparison_statistics.csv', oracle+vlm+stats)
            assert update.publish_human(package)==12
            figures = temp/'latex_auro/figures'
            manifest = json.loads((figures/'00_DATA/experiment_panels_manifest.json').read_text())
            assert len(manifest)==12
            for item in manifest:
                assert (figures/item['file']).read_bytes().startswith(b'%PDF-')
                human = [r for r in item['records'] if r.get('environment')=='Human']
                assert human and {int(r['total_n']) for r in human}=={10}
        finally:
            build.ROOT = actual_root
    assert all(hashlib.sha256(p.read_bytes()).hexdigest()==digest for p,digest in original_files.items())
    print('PASS: 24/32/40 trials, automatic participant counts, 12 published PDFs; real figures unchanged by test.')

if __name__ == '__main__':
    main()
