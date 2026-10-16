"""Compatibility entry point; canonical results live inside asset/BRL_EXP."""
from pathlib import Path
import subprocess
import sys

if __name__ == '__main__':
    script = Path(__file__).resolve().parents[1] / 'asset/BRL_EXP/04_CODE/classify_vlm_failures.py'
    subprocess.run([sys.executable, str(script), *sys.argv[1:]], check=True)
