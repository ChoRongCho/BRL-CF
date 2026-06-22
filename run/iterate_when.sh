#!/usr/bin/env bash

set -euo pipefail

domains=(tomato wastesorting)
# strategies=(all no ours random)
strategies=(random)
scenes=(1 2 3 4 5)

iterations=40
max_step=50
archive_existing=true
log_root="experiments_logs/system_log"

total=$((${#domains[@]} * ${#strategies[@]} * ${#scenes[@]} * iterations))
current=0

printf "\rProgress: %3d%%" 0

timestamp=$(date +%Y%m%d_%H%M%S)

random_query_prob_for_domain() {
    case "$1" in
        tomato)
            printf "0.48"
            ;;
        wastesorting)
            printf "0.38"
            ;;
        *)
            echo "Unknown domain: $1" >&2
            exit 1
            ;;
    esac
}

if [[ "${archive_existing}" == "true" ]]; then
    for domain in "${domains[@]}"; do
        random_query_prob=$(random_query_prob_for_domain "$domain")
        prob_label="${random_query_prob/./-}"
        for strategy in "${strategies[@]}"; do
            for scene in "${scenes[@]}"; do
                scene_id=$(printf "%02d" "$((10#$scene))")
                log_dir="${log_root}/${domain}/scene_${scene_id}_step${max_step}/when_${strategy}_rand_${prob_label}"
                if [[ -d "${log_dir}" ]]; then
                    mv "${log_dir}" "${log_dir}.backup_${timestamp}"
                fi
            done
        done
    done
fi

for domain in "${domains[@]}"; do
    random_query_prob=$(random_query_prob_for_domain "$domain")
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

python3 experiments/system_eval/analysis_experiment.py >/dev/null
python3 experiments/system_eval/read_csv_experiment.py >/dev/null
printf "\rProgress: 100%%\n"
