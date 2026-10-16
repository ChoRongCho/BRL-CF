"""Refresh Oracle panels using BRL_EXP and the shared manuscript figure layout."""
from pathlib import Path
import shutil
import subprocess
import sys

if __name__ == '__main__':
    package = Path(__file__).resolve().parents[1]/'asset/BRL_EXP'
    code = package/'04_CODE'
    subprocess.run([sys.executable, str(code/'build_data.py')], check=True)
    sys.path.insert(0, str(code))
    import plot_experiment_panels as plot
    from figure_paths import publish_panels, ORACLE_DATA_PATH
    plot.style()
    plot.render_oracle(plot.read(plot.BASE/'exp_oracle_baselines_data.csv'))
    publish_panels(package, plot.MANIFEST)
    shutil.copy2(plot.BASE/'exp_oracle_baselines_data.csv', package.parents[1]/'figures'/ORACLE_DATA_PATH)
