"""Rebuild local data, failure classifications and all 33 manuscript panels.

Use --publish-figures to copy newly generated panels into latex_auro/figures.
Source installation is separate: organize_sources.py reads original workspace files.
"""
from pathlib import Path
import argparse
import json
import shutil
import subprocess
import sys

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--publish-figures', action='store_true')
    args = parser.parse_args()
    for script in ['build_data.py', 'classify_vlm_failures.py', 'plot_experiment_panels.py']:
        subprocess.run([sys.executable, str(CODE / script)], check=True)
    if args.publish_figures:
        from figure_paths import publish_panels, ORACLE_DATA_PATH
        target = ROOT.parents[1] / 'figures'
        manifest = json.loads((ROOT/'experiment_panels_manifest.json').read_text())
        publish_panels(ROOT, manifest)
        shutil.copy2(ROOT/'01_BASELINE/02_graph_data/exp_oracle_baselines_data.csv', target/ORACLE_DATA_PATH)
        print(f'Published {len(manifest)} panels to {target}. LaTeX prose/tables are edited separately.')

if __name__ == '__main__':
    main()
