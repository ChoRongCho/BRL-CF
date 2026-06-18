#!/usr/bin/env bash

set -euo pipefail

domains=(tomato wastesorting)
thresholds=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
scenes=(1 2 3 4 5)


iterations=40
seed="${SEED:-random}"
seed_log_root="experiments_logs/system_log/threshold_seed_logs"
seed_log="${seed_log_root}/iterate_th_$(date +%Y%m%d_%H%M%S).csv"

total=$((${#domains[@]} * ${#thresholds[@]} * ${#scenes[@]} * iterations))
current=0

mkdir -p "$seed_log_root"
echo "global_index,domain,threshold,scene,seed_mode" > "$seed_log"
printf "\rProgress: %3d%%" 0

for domain in "${domains[@]}"; do
    for threshold in "${thresholds[@]}"; do
        for scene in "${scenes[@]}"; do
            for ((i = 1; i <= iterations; i++)); do
                current=$((current + 1))
                echo "${current},${domain},${threshold},${scene},${seed}" >> "$seed_log"
                ./run/run_threshold_experiment.sh --domain "$domain" --scene "$scene" --iter 1 --threshold "$threshold" --seed "$seed" >/dev/null
                percent=$((current * 100 / total))
                printf "\rProgress: %3d%%" "$percent"
            done
        done
    done
done

python3 experiments/system_eval/analysis_experiment.py >/dev/null
python3 experiments/system_eval/read_csv_experiment.py >/dev/null
printf "\rProgress: 100%%\n"
echo "Seed mode log saved to ${seed_log}"
