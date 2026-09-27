#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON:-/home/fr/miniconda3/envs/brl/bin/python}"

EPISODES_PER_SCENE="25"
BASE_SEED="2026092101"
COVERAGE="0.95"
N_SIMULATIONS="100"
GAMMA="0.2"
OUTPUT_DIR="$PROJECT_ROOT/scripts/ablation/when_what_policy_ablation/dataset"

while (($#)); do
  case "$1" in
    --episodes-per-scene) EPISODES_PER_SCENE="${2:?Missing value}"; shift 2 ;;
    --base-seed) BASE_SEED="${2:?Missing value}"; shift 2 ;;
    --coverage) COVERAGE="${2:?Missing value}"; shift 2 ;;
    --n-simulations) N_SIMULATIONS="${2:?Missing value}"; shift 2 ;;
    --gamma) GAMMA="${2:?Missing value}"; shift 2 ;;
    --output-dir) OUTPUT_DIR="${2:?Missing value}"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 [--episodes-per-scene 25] [--coverage 0.95] [--output-dir PATH]"
      exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

DATASET="$OUTPUT_DIR/state_cp_dataset.json"
RESULT="$OUTPUT_DIR/state_cp_qhat.json"
CSV="$OUTPUT_DIR/state_cp_records.csv"
mkdir -p "$OUTPUT_DIR"
cd "$PROJECT_ROOT"

"$PYTHON_BIN" -m scripts.ablation.when_what_policy_ablation.script.collect_state_cp_dataset \
  --episodes-per-scene "$EPISODES_PER_SCENE" \
  --base-seed "$BASE_SEED" \
  --gamma "$GAMMA" \
  --n-simulations "$N_SIMULATIONS" \
  --output "$DATASET"

"$PYTHON_BIN" -m scripts.ablation.when_what_policy_ablation.script.calibrate_state_cp \
  --dataset "$DATASET" \
  --coverage "$COVERAGE" \
  --output "$RESULT" \
  --output-csv "$CSV"

echo "Dataset: $DATASET"
echo "Calibration: $RESULT"
echo "Use with: STATE_CALIBRATION=$RESULT ./run/run_when_what_policy_ablation.sh --condition cp_when ..."
