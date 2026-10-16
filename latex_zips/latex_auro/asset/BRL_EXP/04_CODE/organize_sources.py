"""Copy the manuscript's source batches into BRL_EXP; never move originals.
Run once from any directory. Rerunning refreshes source snapshots explicitly.
"""
from pathlib import Path
import csv
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT.parents[1]
CODE = PAPER.parents[1]
PROJECT = CODE.parent
EXP = CODE / 'experiments_final'
SECTIONS = ['01_BASELINE', '02_ABLATION', '03_THRESHOLD']
VIDEO = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v'}

def read(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader(); w.writerows(rows)

COPIES = []

def copy_file(src, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    data = dest.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    assert digest == hashlib.sha256(src.read_bytes()).hexdigest(), src
    COPIES.append(dict(source=str(src.relative_to(PROJECT)), destination=str(dest.relative_to(ROOT)),
                       bytes=len(data), sha256=digest))

def copy_tree(src, dest):
    for f in sorted(src.rglob('*')):
        if f.is_file() and f.suffix.lower() not in VIDEO and f.name != '.DS_Store' and '__pycache__' not in f.parts:
            copy_file(f, dest / f.relative_to(src))

def main():
    for section in SECTIONS:
        for stage in ['00_raw', '01_processed', '02_graph_data', '03_figures']:
            (ROOT / section / stage).mkdir(parents=True, exist_ok=True)
    index = []
    batches = [('01_BASELINE', 'oracle_baselines', '03_baseline'),
               ('02_ABLATION', 'random_when_what', '01_when_what_random'),
               ('02_ABLATION', 'policy_when_what', '02_when_what_policy_ablation'),
               ('03_THRESHOLD', 'threshold_sweep', '00_threshold')]
    for section, batch, package in batches:
        src = EXP / '01_Oracle' / package
        target = ROOT / section / '00_raw' / batch
        copy_tree(src / '00_raw', target)
        inputs = ROOT / section / '00_raw/source_inputs' / batch
        for f in [*src.glob('*.json'), *src.glob('*.csv'), src / 'README.md', *src.glob('01_processed/*.csv')]:
            copy_file(f, inputs / f.name)
        for r in read(src / '01_processed/episodes.csv'):
            raw = target / r['raw_source'].split('/00_raw/', 1)[1]
            assert hashlib.sha256(raw.read_bytes()).hexdigest() == r['raw_sha256'], raw
            index.append({**r, 'batch': batch, 'section': section, 'environment': 'Oracle',
                          'raw_path': str(raw.relative_to(ROOT))})
    vlm = EXP / '02_VLM/00_vlm_exp'
    copy_tree(EXP / '02_VLM/00_vlm_exp/00_raw', ROOT / '01_BASELINE/00_raw/vlm_sessions')
    copy_file(EXP / '02_VLM/00_vlm_exp/01_processed/failures.csv',
              ROOT / '01_BASELINE/00_raw/source_inputs/vlm/session_failures.csv')
    for f in [*vlm.glob('*.csv'), *vlm.glob('01_processed/*.csv')]:
        copy_file(f, ROOT / '01_BASELINE/00_raw/source_inputs/vlm' / f.name)
    summary = EXP / '04_Total_Summary/data'
    for name in ['vlm_waste_source.csv', 'vlm_tomato_source.csv']:
        copy_file(summary / name, ROOT / '01_BASELINE/00_raw/source_inputs/vlm' / name)
        for r in read(summary / name):
            raw = ROOT / '01_BASELINE/00_raw/vlm' / r['domain'] / r['run'] / Path(r['raw_copy']).name
            source = vlm / r['raw_copy']
            copy_file(source, raw)
            assert hashlib.sha256(raw.read_bytes()).hexdigest() == r['raw_sha256'], raw
            index.append({**r, 'batch': 'vlm', 'section': '01_BASELINE', 'environment': 'VLM',
                          'condition': r['method'], 'status': 'ok', 'raw_path': str(raw.relative_to(ROOT))})
    human = PROJECT / '08_Delete_Me/00_real_human'
    for p in sorted(human.glob('p[0-9]*')):
        if p.is_dir():
            copy_tree(p, ROOT / '01_BASELINE/00_raw/human' / p.name)
    for name in ['human_summary.csv', 'human_aggregate.csv']:
        copy_file(human / name, ROOT / '01_BASELINE/00_raw/source_inputs/human' / name)
    # Keep the old manuscript numbers traceable, separate from current human data.
    for f in summary.glob('*.csv'):
        copy_file(f, ROOT / '01_BASELINE/00_raw/source_inputs/manuscript_snapshot' / f.name)
    copy_file(PAPER / 'figures/00_DATA/exp_oracle_baselines_data.csv', ROOT / '01_BASELINE/00_raw/source_inputs/manuscript_snapshot/exp_oracle_baselines_data.csv')
    copy_file(PAPER / 'figures/00_DATA/experiment_panels_manifest.json', ROOT / '01_BASELINE/00_raw/source_inputs/manuscript_snapshot/experiment_panels_manifest.json')
    write(ROOT / 'source_index.csv', index)
    # Retain independently archived input snapshots already installed locally.
    previous = read(ROOT / 'source_files.csv') if (ROOT / 'source_files.csv').exists() else []
    by_destination = {r['destination']: r for r in previous if (ROOT/r['destination']).is_file()}
    by_destination.update({r['destination']: r for r in COPIES})
    write(ROOT / 'source_files.csv', list(by_destination.values()))
    print(json.dumps({'copied_files': len(COPIES), 'indexed_runs': len(index),
                      'bytes': sum(r['bytes'] for r in COPIES)}, indent=2))

if __name__ == '__main__':
    main()
