#!/usr/bin/env bash

set -euo pipefail

# thresholds=(0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
thresholds=(0.8)

scenes=(1 2 3 4 5)
iterations=10

total=$((${#thresholds[@]} * ${#scenes[@]} * iterations))
current=0

printf "\rProgress: %3d%%" 0

for threshold in "${thresholds[@]}"; do
    for scene in "${scenes[@]}"; do
        for ((i = 1; i <= iterations; i++)); do
            ./run/run_threshold_experiment.sh --scene "$scene" --iter 1 --threshold "$threshold" >/dev/null
            current=$((current + 1))
            percent=$((current * 100 / total))
            printf "\rProgress: %3d%%" "$percent"
        done
    done
done

python3 exp_code/analysis_files.py >/dev/null
printf "\rProgress: 100%%\n"
