#!/usr/bin/env bash

set -euo pipefail

domains=(tomato wastesorting)
thresholds=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
scenes=(1 2 3 4 5)


iterations=40
# SEED=random: generate one random seed per domain/scene/iteration pair.
# SEED=N: use N as a reproducible base and derive a distinct seed per pair.
seed_mode="${SEED:-random}"
archive_existing="${ARCHIVE_EXISTING:-true}"
log_root="experiments_logs/system_log"
archive_root="experiments_logs/system_log_backup/threshold_$(date +%Y%m%d_%H%M%S)"
seed_log_root="${log_root}/threshold_seed_logs"
seed_log="${seed_log_root}/iterate_th_$(date +%Y%m%d_%H%M%S).csv"

total=$((${#domains[@]} * ${#thresholds[@]} * ${#scenes[@]} * iterations))
current=0

mkdir -p "$seed_log_root"
echo "global_index,pair_id,domain,scene,iteration,threshold,seed" > "$seed_log"
printf "\rProgress: %3d%%" 0

if [[ "$seed_mode" != "random" && ! "$seed_mode" =~ ^[0-9]+$ ]]; then
    echo "SEED must be a non-negative integer or random: ${seed_mode}"
    exit 1
fi

generate_seed() {
    od -An -N4 -tu4 /dev/urandom | tr -d ' '
}

if [[ "$archive_existing" == "true" ]]; then
    for domain in "${domains[@]}"; do
        for threshold in "${thresholds[@]}"; do
            threshold_label="${threshold/./-}"
            for scene in "${scenes[@]}"; do
                scene_id=$(printf "%02d" "$((10#$scene))")
                log_dir="${log_root}/${domain}/scene_${scene_id}_step50/thres_${threshold_label}"
                if [[ -d "$log_dir" ]]; then
                    archive_dir="${archive_root}/${domain}/scene_${scene_id}_step50"
                    mkdir -p "$archive_dir"
                    mv "$log_dir" "$archive_dir/"
                fi
            done
        done
    done
fi

pair_id=0
for domain_index in "${!domains[@]}"; do
    domain="${domains[$domain_index]}"
    for scene in "${scenes[@]}"; do
        for ((i = 1; i <= iterations; i++)); do
            pair_id=$((pair_id + 1))
            if [[ "$seed_mode" == "random" ]]; then
                paired_seed=$(generate_seed)
            else
                # Distinct and reproducible across domain, scene, and repetition.
                paired_seed=$((
                    (10#$seed_mode + domain_index * 1000000 + 10#$scene * 1000 + i)
                    % 4294967295
                ))
            fi

            for threshold in "${thresholds[@]}"; do
                current=$((current + 1))
                echo "${current},${pair_id},${domain},${scene},${i},${threshold},${paired_seed}" >> "$seed_log"
                ./run/run_threshold_experiment.sh --domain "$domain" --scene "$scene" --iter 1 --threshold "$threshold" --seed "$paired_seed" >/dev/null
                percent=$((current * 100 / total))
                printf "\rProgress: %3d%%" "$percent"
            done
        done
    done
done

python3 experiments/system_eval/analysis_experiment.py >/dev/null
python3 experiments/system_eval/read_csv_experiment.py >/dev/null
printf "\rProgress: 100%%\n"
echo "Paired seed log saved to ${seed_log}"
