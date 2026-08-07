#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
python3 "$SCRIPT_DIR/e1/analysis.py" "$@"
python3 "$SCRIPT_DIR/e1/read.py"
python3 "$SCRIPT_DIR/e1/plot.py"
