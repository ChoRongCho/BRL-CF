#!/usr/bin/env bash

set -euo pipefail

DOMAINS=(tomato wastesorting)
SCENES=(1 2 3 4 5)
ITERATIONS="40"
MAX_STEP="50"
ARCHIVE_EXISTING="true"
LOG_ROOT="experiments_logs/system_log"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
timestamp=$(date +%Y%m%d_%H%M%S)
total=$((${#DOMAINS[@]} * ${#SCENES[@]} * ITERATIONS))
current=0

cd "$PROJECT_ROOT"

if [[ "$ARCHIVE_EXISTING" == "true" ]]; then
    for domain in "${DOMAINS[@]}"; do
        for scene in "${SCENES[@]}"; do
            scene_id=$(printf "%02d" "$((10#$scene))")
            target="${LOG_ROOT}/${domain}/scene_${scene_id}_step${MAX_STEP}/attr_pomdp"
            if [[ -d "$target" ]]; then
                mv "$target" "${target}.backup_${timestamp}"
            fi
        done
    done
fi

seed_dir="${LOG_ROOT}/attr_pomdp_seed_logs"
mkdir -p "$seed_dir"
seed_log="${seed_dir}/iterate_attr_pomdp_${timestamp}.csv"
echo "global_index,domain,scene,iteration,seed" > "$seed_log"

printf "\rProgress: %3d%%" 0
for domain in "${DOMAINS[@]}"; do
    for scene in "${SCENES[@]}"; do
        for ((iteration = 1; iteration <= ITERATIONS; iteration++)); do
            seed=$(od -An -N4 -tu4 /dev/urandom | tr -d ' ')
            current=$((current + 1))
            echo "${current},${domain},${scene},${iteration},${seed}" >> "$seed_log"
            DOMAIN="$domain" \
            SCENE="$scene" \
            SEED="$seed" \
            MAX_STEP="$MAX_STEP" \
            LOG_ROOT="$LOG_ROOT" \
                "$SCRIPT_DIR/run.sh" >/dev/null
            printf "\rProgress: %3d%%" "$((current * 100 / total))"
        done
    done
done

printf "\rProgress: 100%%\n"
echo "Seed log saved to ${seed_log}"
