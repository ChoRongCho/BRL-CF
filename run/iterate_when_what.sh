#!/usr/bin/env bash

set -euo pipefail

# ==================== Experiment configuration ====================
# Edit these globals, then run: ./run/iterate_when_what.sh
DOMAINS=(tomato wastesorting)
CONDITIONS=(random ours-when-only ours-what-only ours)
SCENES=(1 2 3 4 5)

ITERATIONS="40"
MAXSTEP="50"

# Use one shared Random-When probability for both domains.
THRESHOLD="0.8"
TOMATO_RANDOM_QUERY_PROB="0.4"
WASTE_RANDOM_QUERY_PROB="0.4"

LOG_ROOT="experiments_logs/system_log"
ARCHIVE_EXISTING="true"
# ==================================================================

usage() {
    echo "Usage: $0 [options]"
    echo "With no options, values from the configuration block at the top are used."
    echo "  --threshold T     confidence threshold (default: ${THRESHOLD})"
    echo "  --tomato-random-query-prob P      (default: ${TOMATO_RANDOM_QUERY_PROB})"
    echo "  --waste-random-query-prob P       (default: ${WASTE_RANDOM_QUERY_PROB})"
    echo "  --iter N          repetitions per domain/condition/scene (default: ${ITERATIONS})"
    echo "  --max-step N      maximum episode steps (default: ${MAXSTEP})"
    echo "  --log-root PATH   log root (default: ${LOG_ROOT})"
    echo "  --no-archive      keep existing target log directories"
}

while (($#)); do
    case "$1" in
        --threshold) THRESHOLD="${2:?Missing value for --threshold}"; shift 2 ;;
        --tomato-random-query-prob) TOMATO_RANDOM_QUERY_PROB="${2:?Missing value for --tomato-random-query-prob}"; shift 2 ;;
        --waste-random-query-prob) WASTE_RANDOM_QUERY_PROB="${2:?Missing value for --waste-random-query-prob}"; shift 2 ;;
        --iter|--iteration) ITERATIONS="${2:?Missing value for $1}"; shift 2 ;;
        --max-step) MAXSTEP="${2:?Missing value for --max-step}"; shift 2 ;;
        --log-root) LOG_ROOT="${2:?Missing value for --log-root}"; shift 2 ;;
        --no-archive) ARCHIVE_EXISTING="false"; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1"; usage; exit 1 ;;
    esac
done

if ! [[ "$ITERATIONS" =~ ^[1-9][0-9]*$ && "$MAXSTEP" =~ ^[1-9][0-9]*$ ]]; then
    echo "iterations and max-step must be positive integers."
    exit 1
fi

python3 - "$THRESHOLD" "$TOMATO_RANDOM_QUERY_PROB" "$WASTE_RANDOM_QUERY_PROB" <<'PY'
import sys
for name, value in zip(("threshold", "tomato probability", "waste probability"), sys.argv[1:]):
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise SystemExit(f"{name} must be between 0 and 1: {value}")
PY

total=$((${#DOMAINS[@]} * ${#CONDITIONS[@]} * ${#SCENES[@]} * ITERATIONS))
current=0
timestamp=$(date +%Y%m%d_%H%M%S)
threshold_label=${THRESHOLD/./-}

if [[ "$ARCHIVE_EXISTING" == "true" ]]; then
    for domain in "${DOMAINS[@]}"; do
        if [[ "$domain" == "tomato" ]]; then
            probability="$TOMATO_RANDOM_QUERY_PROB"
        else
            probability="$WASTE_RANDOM_QUERY_PROB"
        fi
        prob_label=${probability/./-}
        for condition in "${CONDITIONS[@]}"; do
            condition_label=${condition//-/_}
            for scene in "${SCENES[@]}"; do
                scene_id=$(printf "%02d" "$((10#$scene))")
                target="${LOG_ROOT}/${domain}/scene_${scene_id}_step${MAXSTEP}/when_what_${condition_label}_thres_${threshold_label}_rand_${prob_label}"
                if [[ -d "$target" ]]; then
                    mv "$target" "${target}.backup_${timestamp}"
                fi
            done
        done
    done
fi

seed_log_dir="${LOG_ROOT}/when_what_seed_logs"
mkdir -p "$seed_log_dir"
seed_log="${seed_log_dir}/iterate_when_what_${timestamp}.csv"
echo "domain,condition,scene,iteration,seed,threshold,random_query_prob" > "$seed_log"

printf "\rProgress: %3d%%" 0
for domain in "${DOMAINS[@]}"; do
    if [[ "$domain" == "tomato" ]]; then
        probability="$TOMATO_RANDOM_QUERY_PROB"
    else
        probability="$WASTE_RANDOM_QUERY_PROB"
    fi
    for scene in "${SCENES[@]}"; do
        for ((iteration = 1; iteration <= ITERATIONS; iteration++)); do
            # Pair all four conditions with the same episode seed.
            seed=$(od -An -N4 -tu4 /dev/urandom | tr -d ' ')
            for condition in "${CONDITIONS[@]}"; do
                echo "${domain},${condition},${scene},${iteration},${seed},${THRESHOLD},${probability}" >> "$seed_log"
                ./run/run_when_what_ablation.sh \
                    --domain "$domain" \
                    --scene "$scene" \
                    --iter 1 \
                    --condition "$condition" \
                    --threshold "$THRESHOLD" \
                    --random-query-prob "$probability" \
                    --max-step "$MAXSTEP" \
                    --seed "$seed" \
                    --log-root "$LOG_ROOT" >/dev/null
                current=$((current + 1))
                printf "\rProgress: %3d%%" "$((current * 100 / total))"
            done
        done
    done
done

printf "\rProgress: 100%%\n"
echo "Seed log saved to ${seed_log}"
