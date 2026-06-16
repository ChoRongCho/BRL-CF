#!/usr/bin/env bash

set -euo pipefail

domains=(tomato wastesorting)
strategies=(random)
scenes=(1 2 3 4 5)

iterations=40
max_step=50
random_query_prob="0.3"
archive_existing=true

total=$((${#domains[@]} * ${#strategies[@]} * ${#scenes[@]} * iterations))
current=0

printf "\rProgress: %3d%%" 0

timestamp=$(date +%Y%m%d_%H%M%S)
prob_label="${random_query_prob/./-}"

if [[ "${archive_existing}" == "true" ]]; then
    for domain in "${domains[@]}"; do
        for strategy in "${strategies[@]}"; do
            for scene in "${scenes[@]}"; do
                scene_id=$(printf "%02d" "$((10#$scene))")
                log_dir="logs/${domain}/scene_${scene_id}_step${max_step}/when_${strategy}_rand_${prob_label}"
                if [[ -d "${log_dir}" ]]; then
                    mv "${log_dir}" "${log_dir}.backup_${timestamp}"
                fi
            done
        done
    done
fi

for domain in "${domains[@]}"; do
    for strategy in "${strategies[@]}"; do
        for scene in "${scenes[@]}"; do
            for ((i = 1; i <= iterations; i++)); do
                ./run/when_experiments.sh \
                    --domain "$domain" \
                    --scene "$scene" \
                    --iter 1 \
                    --strategy "$strategy" \
                    --random-query-prob "$random_query_prob" \
                    --max-step "$max_step" \
                    --seed random >/dev/null
                current=$((current + 1))
                percent=$((current * 100 / total))
                printf "\rProgress: %3d%%" "$percent"
            done
        done
    done
done

python3 exp_code/analysis_when.py >/dev/null
python3 exp_code/read_csv_when_bar_v2.py >/dev/null
printf "\rProgress: 100%%\n"
