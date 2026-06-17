#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BASELINE_DIR="${PROJECT_ROOT}/scripts/baseline"

cd "${PROJECT_ROOT}"
python3 "${BASELINE_DIR}/run_knowno_baseline_gui.py"
