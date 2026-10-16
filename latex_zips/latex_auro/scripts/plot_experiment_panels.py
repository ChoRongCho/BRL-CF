"""Compatibility entry point for the canonical BRL_EXP graph pipeline."""
from pathlib import Path
import subprocess
import sys

if __name__ == '__main__':
    script = Path(__file__).resolve().parents[1]/'asset/BRL_EXP/04_CODE/rebuild.py'
    subprocess.run([sys.executable, str(script), '--publish-figures', *sys.argv[1:]], check=True)
