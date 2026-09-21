#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

python_is_compatible() {
  "$1" -c 'import sys; raise SystemExit(sys.version_info < (3, 10))' >/dev/null 2>&1
}

if [[ -n "${PYTHON:-}" ]]; then
  PYTHON_BIN="$PYTHON"
  python_is_compatible "$PYTHON_BIN" || {
    echo "PYTHON must point to Python 3.10 or newer: $PYTHON_BIN" >&2
    exit 1
  }
elif python_is_compatible python3; then
  PYTHON_BIN="python3"
elif [[ -x /home/fr/miniconda3/envs/brl/bin/python ]] && python_is_compatible /home/fr/miniconda3/envs/brl/bin/python; then
  PYTHON_BIN="/home/fr/miniconda3/envs/brl/bin/python"
else
  echo "Python 3.10 or newer is required. Activate the brl environment or set PYTHON." >&2
  exit 1
fi
export PYTHON="$PYTHON_BIN"

export WW_CONDITIONS="${WW_CONDITIONS:-ours cp_when value_when value_what}"
export WW_DOMAINS="${WW_DOMAINS:-tomato wastesorting}"
export WW_SCENES="${WW_SCENES:-1 2 3 4 5}"
export WW_ITERATIONS="${WW_ITERATIONS:-40}"
export WW_PAIRED_SEED_LOG="${WW_PAIRED_SEED_LOG:-experiments_logs/system_log/when_what_seed_logs/iterate_when_what_20260912_150132.csv}"
export WW_LOG_ROOT="${WW_LOG_ROOT:-experiments_logs/when_what_mechanisms}"
export WW_RUN_ROOT="${WW_RUN_ROOT:-}"
export WW_DRY_RUN="${WW_DRY_RUN:-false}"
export WW_RESUME="${WW_RESUME:-false}"

while (($#)); do
  case "$1" in
    --condition) WW_CONDITIONS="${2:?Missing condition}"; export WW_CONDITIONS; shift 2 ;;
    --conditions) WW_CONDITIONS="${2:?Missing conditions}"; export WW_CONDITIONS; shift 2 ;;
    --iter|--iteration) WW_ITERATIONS="${2:?Missing iterations}"; export WW_ITERATIONS; shift 2 ;;
    --dry-run) WW_DRY_RUN="true"; export WW_DRY_RUN; shift ;;
    --resume) WW_RESUME="true"; export WW_RESUME; shift ;;
    --run-root) WW_RUN_ROOT="${2:?Missing run root}"; export WW_RUN_ROOT; shift 2 ;;
    -h|--help)
      echo "Usage: $0 [--condition NAME|--conditions \"NAMES\"] [--iter N] [--dry-run] [--resume --run-root PATH]"
      exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

cd "$PROJECT_ROOT"
exec "$PYTHON_BIN" scripts/ablation/when_what_mechanisms/batch.py
