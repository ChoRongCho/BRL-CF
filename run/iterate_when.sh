#!/usr/bin/env bash

set -euo pipefail

strategies=(no all random ours)
# scenes=(1 2 3 4 5)
scenes=(5)

iterations=40

total=$((${#strategies[@]} * ${#scenes[@]} * iterations))
current=0

printf "\rProgress: %3d%%" 0

for strategy in "${strategies[@]}"; do
    for scene in "${scenes[@]}"; do
        for ((i = 1; i <= iterations; i++)); do
            ./run/when_experiments.sh --scene "$scene" --iter 1 --strategy "$strategy" >/dev/null
            current=$((current + 1))
            percent=$((current * 100 / total))
            printf "\rProgress: %3d%%" "$percent"
        done
    done
done

python3 exp_code/analysis_when.py >/dev/null
printf "\rProgress: 100%%\n"
