#!/usr/bin/env bash

set -euo pipefail

# domains=(tomato wastesorting)
# scenes=(1 2 3 4 5)
domains=(wastesorting)
scenes=(1)



iterations="${ITERATIONS:-40}"
threshold="${THRESHOLD:-0.8}"
seed="${SEED:-random}"
archive_existing="${ARCHIVE_EXISTING:-true}"
log_root="experiments_logs/system_log"
seed_log_root="${log_root}/scale_seed_logs"
seed_log="${seed_log_root}/iterate_scale_$(date +%Y%m%d_%H%M%S).csv"

total=$((${#domains[@]} * ${#scenes[@]} * iterations))
current=0
timestamp=$(date +%Y%m%d_%H%M%S)

mkdir -p "$seed_log_root"
echo "global_index,domain,scene,threshold,seed_mode,max_step,max_belief_particles" > "$seed_log"

max_step_for_scene() {
    local scene="$1"
    if (( scene >= 1 && scene <= 5 )); then
        echo "50"
    elif (( scene >= 6 && scene <= 10 )); then
        echo "90"
    elif (( scene >= 11 && scene <= 15 )); then
        echo "125"
    else
        echo "50"
    fi
}

max_belief_particles_for_scene() {
    local scene="$1"
    if (( scene >= 1 && scene <= 5 )); then
        echo "800"
    elif (( scene >= 6 && scene <= 10 )); then
        echo "800"
    elif (( scene >= 11 && scene <= 15 )); then
        echo "8000"
    else
        echo "8000"
    fi
}

for domain in "${domains[@]}"; do
    for scene in "${scenes[@]}"; do
        scene_id=$(printf "%02d" "$((10#$scene))")
        max_step=$(max_step_for_scene "$scene")
        printf "Target output dir: %s\n" "${log_root}/${domain}/scene_${scene_id}/scale_ours_step${max_step}"
    done
done

printf "\rProgress: %3d%%" 0

if [[ "$archive_existing" == "true" ]]; then
    for domain in "${domains[@]}"; do
        for scene in "${scenes[@]}"; do
            scene_id=$(printf "%02d" "$((10#$scene))")
            max_step=$(max_step_for_scene "$scene")
            max_belief_particles=$(max_belief_particles_for_scene "$scene")
            log_dir="${log_root}/${domain}/scene_${scene_id}/scale_ours_step${max_step}"
            if [[ -d "$log_dir" ]]; then
                mv "$log_dir" "${log_dir}.backup_${timestamp}"
            fi
        done
    done
fi

for domain in "${domains[@]}"; do
    for scene in "${scenes[@]}"; do
        max_step=$(max_step_for_scene "$scene")
        max_belief_particles=$(max_belief_particles_for_scene "$scene")
        for ((i = 1; i <= iterations; i++)); do
            current=$((current + 1))
            echo "${current},${domain},${scene},${threshold},${seed},${max_step},${max_belief_particles}" >> "$seed_log"
            ./run/run_scale_experiments.sh \
                --domain "$domain" \
                --scene "$scene" \
                --iter 1 \
                --threshold "$threshold" \
                --seed "$seed" \
                --max-step "$max_step" \
                --max-belief-particles "$max_belief_particles" >/dev/null
            percent=$((current * 100 / total))
            printf "\rProgress: %3d%%" "$percent"
        done
    done
done

printf "\rProgress: 100%%\n"
echo "Seed mode log saved to ${seed_log}"
