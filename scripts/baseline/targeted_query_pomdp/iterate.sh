#!/usr/bin/env bash

# Compatibility entry point. Both query baselines are configured and paired
# by run/iterate_query_baselines.sh.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/../../../run/iterate_query_baselines.sh" --baseline query_action_pomcp "$@"

: <<'LEGACY_IMPLEMENTATION'

# ==================== Experiment configuration ====================
DOMAINS=(tomato wastesorting)
SCENES=(1 2 3 4 5)
ITERATIONS="40"
MAX_STEP="50"
N_SIMULATIONS="100"
MAX_DEPTH="20"
QUERY_COST="1.0"
ANSWER_ACCURACY="1.0"
LOG_ROOT="experiments_logs/system_log"
ARCHIVE_EXISTING="true"
PAIRED_SEED_LOG="experiments_logs/system_log/when_what_seed_logs/iterate_when_what_20260912_150132.csv"
# ==================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
timestamp=$(date +%Y%m%d_%H%M%S)
total=$((${#DOMAINS[@]} * ${#SCENES[@]} * ITERATIONS))
current=0

cd "$PROJECT_ROOT"

if [[ ! -f "$PAIRED_SEED_LOG" ]]; then
    echo "Paired seed log not found: ${PAIRED_SEED_LOG}"
    exit 1
fi

if [[ "$ARCHIVE_EXISTING" == "true" ]]; then
    for domain in "${DOMAINS[@]}"; do
        for scene in "${SCENES[@]}"; do
            scene_id=$(printf "%02d" "$((10#$scene))")
            target="${LOG_ROOT}/${domain}/scene_${scene_id}_step${MAX_STEP}/query_as_action"
            if [[ -d "$target" ]]; then
                mv "$target" "${target}.backup_${timestamp}"
            fi
        done
    done
fi

seed_dir="${LOG_ROOT}/query_as_action_seed_logs"
mkdir -p "$seed_dir"
seed_log="${seed_dir}/iterate_query_as_action_${timestamp}.csv"
echo "global_index,domain,scene,iteration,seed,query_cost,answer_accuracy,n_simulations,max_depth" > "$seed_log"

printf "\rProgress: %3d%%" 0
while IFS=, read -r domain condition scene iteration seed threshold random_query_prob; do
    if [[ "$domain" == "domain" || "$condition" != "ours" ]]; then
        continue
    fi
    if ((iteration > ITERATIONS)); then
        continue
    fi

    selected_domain="false"
    for configured_domain in "${DOMAINS[@]}"; do
        if [[ "$domain" == "$configured_domain" ]]; then
            selected_domain="true"
            break
        fi
    done
    [[ "$selected_domain" == "true" ]] || continue

    selected_scene="false"
    for configured_scene in "${SCENES[@]}"; do
        if [[ "$scene" == "$configured_scene" ]]; then
            selected_scene="true"
            break
        fi
    done
    [[ "$selected_scene" == "true" ]] || continue

    current=$((current + 1))
    echo "${current},${domain},${scene},${iteration},${seed},${QUERY_COST},${ANSWER_ACCURACY},${N_SIMULATIONS},${MAX_DEPTH}" >> "$seed_log"
    DOMAIN="$domain" \
    SCENE="$scene" \
    SEED="$seed" \
    MAX_STEP="$MAX_STEP" \
    N_SIMULATIONS="$N_SIMULATIONS" \
    MAX_DEPTH="$MAX_DEPTH" \
    QUERY_COST="$QUERY_COST" \
    ANSWER_ACCURACY="$ANSWER_ACCURACY" \
    LOG_ROOT="$LOG_ROOT" \
        "$SCRIPT_DIR/run.sh" >/dev/null
    printf "\rProgress: %3d%%" "$((current * 100 / total))"
done < "$PAIRED_SEED_LOG"

if ((current != total)); then
    echo
    echo "Expected ${total} paired seeds but executed ${current}."
    exit 1
fi

printf "\rProgress: 100%%\n"
echo "Paired seed log saved to ${seed_log}"
LEGACY_IMPLEMENTATION
