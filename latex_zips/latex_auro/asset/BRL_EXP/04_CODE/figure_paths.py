"""Shared manuscript figure layout and publication helpers."""
from pathlib import Path
import json
import shutil

DATA_DIR = '00_DATA'
MANIFEST_PATH = Path(DATA_DIR)/'experiment_panels_manifest.json'
ORACLE_DATA_PATH = Path(DATA_DIR)/'exp_oracle_baselines_data.csv'
STATIC = {
    'exp_human_interaction_time_per_query_all.pdf': '',
    'fig_pomdp_preliminaries.pdf': '',
    'fig2_method.pdf': '', 'fig3_implementation.pdf': '',
    '01_WasteSortingEnv.pdf': '03_SETUP', '02_TomatoHarvestEnv.pdf': '03_SETUP',
    'WasteSortingEnv.jpg': '',
    'exp_human_ui.pdf': '', 'human_tablet_interface.jpg': '',
}
PREFIXES = {
    'exp_oracle_': '04_ORACLE_BASELINE', 'exp_vlm_': '05_VLM_BASELINE',
    'exp_human_': '06_HUMAN_BASELINE', 'exp_response_': '07_RESPONSE_COMPARISON',
    'exp_ablation_': '08_ABLATION', 'exp_threshold_': '09_THRESHOLD',
}


def relative_path(filename):
    name = Path(filename).name
    if name == MANIFEST_PATH.name: return MANIFEST_PATH
    if name == ORACLE_DATA_PATH.name: return ORACLE_DATA_PATH
    folder = STATIC.get(name)
    if folder is None:
        folder = next((folder for prefix, folder in PREFIXES.items() if name.startswith(prefix)), '99_LEGACY')
    return Path(folder)/name


def publish_panels(package, generated):
    """Copy panels into their manuscript groups and merge the data manifest."""
    figures = package.parents[1]/'figures'
    manifest_file = figures/MANIFEST_PATH
    previous = json.loads(manifest_file.read_text()) if manifest_file.exists() else []
    published = []
    for item in generated:
        src = package/item['file']
        relative = relative_path(src.name)
        target = figures/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
        published.append({**item, 'file': str(relative)})
    names = {Path(item['file']).name for item in published}
    merged = [item for item in previous if Path(item['file']).name not in names]+published
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    manifest_file.write_text(json.dumps(merged, ensure_ascii=False, indent=2)+'\n')
    return published
