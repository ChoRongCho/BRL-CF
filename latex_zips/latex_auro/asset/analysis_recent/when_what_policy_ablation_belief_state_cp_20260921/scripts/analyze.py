"""Use the shared pipeline for this experiment package."""
from pathlib import Path
import subprocess
import sys
package=Path(__file__).resolve().parents[1]
subprocess.run([sys.executable, str(package.parent/"scripts/pipeline.py"), "--package", package.name, "--stage", "1"], check=True)
