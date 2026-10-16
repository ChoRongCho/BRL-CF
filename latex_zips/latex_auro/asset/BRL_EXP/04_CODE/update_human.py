"""Refresh current human raw/CSVs and publish Human + response comparison PDFs.

Called automatically by 08_Delete_Me/00_real_human/aggregate_human_summary.py.
No fixed participant count: every nonempty pNNN folder is included.
"""
from pathlib import Path
from collections import Counter
import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile

import summarize_human
from build_data import read, write

ROOT = Path(__file__).resolve().parents[1]
VIDEO = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v'}


def sync_human(source, package=ROOT):
    """Validate the new summary before refreshing copies and their hash inventory."""
    actual = [{k: summarize_human.display(r[k]) for k in summarize_human.COLUMNS}
              for r in summarize_human.collect_rows(source)]
    supplied = read(source/'human_summary.csv')
    def key(row):
        return tuple(row[k] for k in summarize_human.COLUMNS)
    if not actual or Counter(map(key, actual)) != Counter(map(key, supplied)):
        raise ValueError('human_summary.csv differs from current logs; run summarize_human.py first.')
    import aggregate_human_summary as aggregate
    expected = [{k: aggregate.display(r[k]) for k in aggregate.COLUMNS}
                for r in aggregate.aggregate(aggregate.read_rows(source/'human_summary.csv'))]
    if read(source/'human_aggregate.csv') != expected:
        raise ValueError('human_aggregate.csv differs from summary; run aggregate_human_summary.py first.')
    prefix = '01_BASELINE/00_raw/human/'
    inputs = '01_BASELINE/00_raw/source_inputs/human/'
    records = [r for r in read(package/'source_files.csv')
               if not r['destination'].startswith((prefix, inputs))]
    raw = package/'01_BASELINE/00_raw'
    staging = Path(tempfile.mkdtemp(prefix='.human-refresh-', dir=raw))
    try:
        pending = []
        participants = sorted(p for p in source.iterdir()
                              if p.is_dir() and re.fullmatch(r'p[0-9]+', p.name))
        for participant in participants:
            for src in sorted(participant.rglob('*')):
                if src.is_file() and src.suffix.lower() not in VIDEO and src.name != '.DS_Store' and '__pycache__' not in src.parts:
                    pending.append((src, prefix + str(src.relative_to(source))))
        pending += [(source/name, inputs+name) for name in ['human_summary.csv','human_aggregate.csv']]
        for src, destination in pending:
            relative = destination.removeprefix('01_BASELINE/00_raw/')
            dest = staging/relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            data = dest.read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            if digest != hashlib.sha256(src.read_bytes()).hexdigest():
                raise ValueError(f'Source changed while copying: {src}')
            project = ROOT.parents[4]
            try: original = str(src.relative_to(project))
            except ValueError: original = str(src)
            records.append(dict(source=original, destination=destination, bytes=len(data), sha256=digest))
        # Replace only our generated Human copies, preventing stale trials after corrections.
        for relative in ['human', 'source_inputs/human']:
            dest = raw/relative
            if dest.exists(): shutil.rmtree(dest)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(staging/relative), str(dest))
        write(package/'source_files.csv', records)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return len(actual)


def publish_human(package=ROOT):
    import plot_experiment_panels as plot
    plot.ROOT = package
    plot.BASE = package/'01_BASELINE/02_graph_data'
    plot.ABLATION = package/'02_ABLATION/02_graph_data'
    plot.THRESHOLD = package/'03_THRESHOLD/02_graph_data'
    plot.MANIFEST.clear()
    plot.style()
    plot.render_physical(environments=['Human'])
    plot.render_response_comparison()
    plot.render_interaction_per_query_all()
    generated = list(plot.MANIFEST)
    assert len(generated) == 12
    local = package/'experiment_panels_manifest.json'
    previous = json.loads(local.read_text()) if local.exists() else []
    names = {Path(r['file']).name for r in generated}
    merged = [r for r in previous if Path(r['file']).name not in names] + generated
    local.write_text(json.dumps(merged, ensure_ascii=False, indent=2)+'\n')
    from figure_paths import publish_panels
    publish_panels(package, generated)
    return len(generated)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path,
                        default=ROOT.parents[4]/'08_Delete_Me/00_real_human')
    args = parser.parse_args()
    count = sync_human(args.source_root.resolve())
    subprocess.run([sys.executable, str(Path(__file__).with_name('build_data.py'))], check=True)
    panels = publish_human()
    print(f'Updated {count} human trials; published {panels} PDFs to latex_auro/figures.')

if __name__ == '__main__':
    main()
