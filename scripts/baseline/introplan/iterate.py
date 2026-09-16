"""Compatibility entry point; settings now live in run/iterate_baseline.sh."""
import os
from pathlib import Path
import sys
runner = Path(__file__).resolve().parents[3] / 'run/iterate_baseline.sh'
if __name__ == '__main__':
    os.execv('/bin/bash', ['bash', str(runner), '--baseline', 'introplan', *sys.argv[1:]])
