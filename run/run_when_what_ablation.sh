#!/usr/bin/env bash

set -euo pipefail

DOMAIN="tomato"
SCENE="1"
ITERATIONS="1"
MAXSTEP="50"
CONDITION="ours"
RANDOM_QUERY_PROB="0.5"
THRESHOLD="0.8"
SEED="random"
LOG_ROOT="experiments_logs/system_log"

usage() {
    echo "Usage: $0 --condition random|ours-when-only|ours-what-only|ours [options]"
    echo "  --domain tomato|wastesorting      (default: ${DOMAIN})"
    echo "  --scene N                         (default: ${SCENE})"
    echo "  --iter N                          (default: ${ITERATIONS})"
    echo "  --threshold T                     (default: ${THRESHOLD})"
    echo "  --random-query-prob P             (default: ${RANDOM_QUERY_PROB})"
    echo "  --max-step N                      (default: ${MAXSTEP})"
    echo "  --seed N|random                   (default: ${SEED})"
    echo "  --log-root PATH                   (default: ${LOG_ROOT})"
}

while (($#)); do
    case "$1" in
        --condition) CONDITION="${2:?Missing value for --condition}"; shift 2 ;;
        --domain) DOMAIN="${2:?Missing value for --domain}"; shift 2 ;;
        --scene) SCENE="${2:?Missing value for --scene}"; shift 2 ;;
        --iter|--iteration) ITERATIONS="${2:?Missing value for $1}"; shift 2 ;;
        --threshold) THRESHOLD="${2:?Missing value for --threshold}"; shift 2 ;;
        --random-query-prob) RANDOM_QUERY_PROB="${2:?Missing value for --random-query-prob}"; shift 2 ;;
        --max-step) MAXSTEP="${2:?Missing value for --max-step}"; shift 2 ;;
        --seed) SEED="${2:?Missing value for --seed}"; shift 2 ;;
        --log-root) LOG_ROOT="${2:?Missing value for --log-root}"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1"; usage; exit 1 ;;
    esac
done

case "$CONDITION" in
    random|ours-when-only|ours-what-only|ours) ;;
    *) echo "Unknown condition: ${CONDITION}"; usage; exit 1 ;;
esac
case "$DOMAIN" in
    tomato|wastesorting) ;;
    *) echo "Unknown domain: ${DOMAIN}"; usage; exit 1 ;;
esac
if ! [[ "$SCENE" =~ ^[0-9]+$ && "$ITERATIONS" =~ ^[1-9][0-9]*$ && "$MAXSTEP" =~ ^[1-9][0-9]*$ ]]; then
    echo "scene, iterations, and max-step must be positive integers."
    exit 1
fi
if [[ "$SEED" != "random" && ! "$SEED" =~ ^[0-9]+$ ]]; then
    echo "seed must be a non-negative integer or random."
    exit 1
fi

python3 - "$THRESHOLD" "$RANDOM_QUERY_PROB" <<'PY'
import sys
for name, value in (("threshold", sys.argv[1]), ("random-query-prob", sys.argv[2])):
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise SystemExit(f"{name} must be between 0 and 1: {value}")
PY

scene_id=$(printf "%02d" "$((10#$SCENE))")
initial_state="scripts/domain/${DOMAIN}/scene_${scene_id}.yaml"
domain_rule="scripts/domain/${DOMAIN}/domain_rule.yaml"
robot_skill="scripts/domain/${DOMAIN}/robot_skill.yaml"
for input_file in "$initial_state" "$domain_rule" "$robot_skill"; do
    if [[ ! -f "$input_file" ]]; then
        echo "Required file not found: ${input_file}"
        exit 1
    fi
done

condition_label=${CONDITION//-/_}
threshold_label=${THRESHOLD/./-}
prob_label=${RANDOM_QUERY_PROB/./-}
log_dir="${LOG_ROOT}/${DOMAIN}/scene_${scene_id}_step${MAXSTEP}/when_what_${condition_label}_thres_${threshold_label}_rand_${prob_label}"
mkdir -p "$log_dir"

generate_seed() {
    od -An -N4 -tu4 /dev/urandom | tr -d ' '
}

for ((run_index = 1; run_index <= ITERATIONS; run_index++)); do
    if [[ "$SEED" == "random" ]]; then
        run_seed=$(generate_seed)
    else
        run_seed="$SEED"
    fi
    echo "[RUN ${run_index}/${ITERATIONS}] condition=${CONDITION}, domain=${DOMAIN}, scene=${scene_id}, threshold=${THRESHOLD}, random_query_prob=${RANDOM_QUERY_PROB}, seed=${run_seed}"
    python3 when_what_ablation_main.py \
        --ablation-condition "$CONDITION" \
        --domain "$DOMAIN" \
        --domain_rule "$domain_rule" \
        --initial_state "$initial_state" \
        --robot_skill "$robot_skill" \
        --threshold "$THRESHOLD" \
        --random_query_prob "$RANDOM_QUERY_PROB" \
        --answer_type auto \
        --seed "$run_seed" \
        --log_dir "$log_dir" \
        --max_step "$MAXSTEP"
done

echo "[DONE] Logs saved under ${log_dir}"
