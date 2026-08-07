#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
python3 "$SCRIPT_DIR/e2/analysis.py" "$@"
python3 "$SCRIPT_DIR/e2/read.py"
python3 "$SCRIPT_DIR/e2/plot.py"
