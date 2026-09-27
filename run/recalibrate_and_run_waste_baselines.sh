#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ITERATIONS="40"
STAMP="$(date +%Y%m%d_%H%M%S_%N)"
CAL_DIR="experiments_logs/calibration/waste_knowno_introplan_${STAMP}"
RUN_ROOT="experiments_logs/baseline_waste_rerun/${STAMP}"

usage() {
    echo "Usage: $0 [--iter N] [--cal-dir PATH] [--run-root PATH]"
}

while (($#)); do
    case "$1" in
        --iter|--iteration) ITERATIONS="${2:?Missing iteration count}"; shift 2 ;;
        --cal-dir) CAL_DIR="${2:?Missing calibration directory}"; shift 2 ;;
        --run-root) RUN_ROOT="${2:?Missing run root}"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
    esac
done

mkdir -p "$CAL_DIR" "$RUN_ROOT"

echo "Calibration output: $ROOT/$CAL_DIR"
echo "Experiment output:  $ROOT/$RUN_ROOT"
echo "Waste episodes:     $((2 * 5 * ITERATIONS))"

python3 scripts/baseline/knowno/compute_qhat.py \
    --domain wastesorting \
    --score-with-llm \
    --regenerate-options \
    --num-calibration 100 \
    --num-test 0 \
    --target-success 0.95 \
    --temperature 5.0 \
    --quantile-method legacy_higher \
    --output-json "$CAL_DIR/knowno_wastesorting.json" \
    --output-csv "$CAL_DIR/knowno_wastesorting.csv" \
    2>&1 | tee "$CAL_DIR/knowno_calibration.log"

python3 scripts/baseline/introplan/compute_qhat.py \
    --domain wastesorting \
    --records "$CAL_DIR/knowno_wastesorting.json" \
    --knowledge scripts/baseline/introplan/knowledge.json \
    --top-k 3 \
    --temperature 5.0 \
    --target-success 0.95 \
    --quantile-method legacy_higher \
    --output "$CAL_DIR/introplan_wastesorting.json" \
    --output-csv "$CAL_DIR/introplan_wastesorting.csv" \
    2>&1 | tee "$CAL_DIR/introplan_calibration.log"

KNOWNO_QHAT="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["qhat"])' "$CAL_DIR/knowno_wastesorting.json")"
INTROPLAN_QHAT="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["qhat"])' "$CAL_DIR/introplan_wastesorting.json")"

cat > "$CAL_DIR/selected_qhats.txt" <<EOF
knowno_wastesorting_qhat=$KNOWNO_QHAT
introplan_wastesorting_qhat=$INTROPLAN_QHAT
target_success=0.95
quantile_method=legacy_higher
score_temperature=5.0
introplan_top_k=3
EOF

echo "KnowNo Waste qhat:    $KNOWNO_QHAT"
echo "IntroPlan Waste qhat: $INTROPLAN_QHAT"

exec ./run/iterate_baseline.sh \
    --baselines "knowno introplan" \
    --domains wastesorting \
    --iter "$ITERATIONS" \
    --knowno-waste-qhat "$KNOWNO_QHAT" \
    --introplan-waste-qhat "$INTROPLAN_QHAT" \
    --log-root "$RUN_ROOT"
