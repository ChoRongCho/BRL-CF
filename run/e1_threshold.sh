#!/usr/bin/env bash
set -euo pipefail

# E1: Ours + Oracle, sweeping the When threshold.
source "$(dirname "$0")/common.sh"

# E1 global settings. Environment variables override these defaults.
E1_ITERATIONS="${E1_ITERATIONS:-40}"
# E1_DOMAINS="${E1_DOMAINS:-tomato}"
E1_DOMAINS="${E1_DOMAINS:-tomato wastesorting}"

# E1_SCENES="${E1_SCENES:-1}"
E1_SCENES="${E1_SCENES:-1 2 3 4 5}"

# THRESHOLDS="${THRESHOLDS:-1.0}"
THRESHOLDS="${THRESHOLDS:-0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0}"

total_runs=$(($(word_count "$E1_DOMAINS") * $(word_count "$E1_SCENES") * $(word_count "$THRESHOLDS") * E1_ITERATIONS))
completed_runs=0
progress_update "$completed_runs" "$total_runs" "E1 threshold"

cd "$ROOT"
for domain in $E1_DOMAINS; do
    for scene in $E1_SCENES; do
        id="$(scene_id "$scene")"
        for threshold in $THRESHOLDS; do
            label="${threshold/./-}"
            for ((run_index = 1; run_index <= E1_ITERATIONS; run_index++)); do
                run_seed="$(experiment_seed "$domain" "$scene" "$run_index")"
                log_dir="$(experiment_log_dir e1_threshold "$domain" "$scene" ours oracle)/tau_$label"
                run_ours "$domain" "$scene" "$threshold" oracle "$run_seed" "$log_dir"
                completed_runs=$((completed_runs + 1))
                progress_update "$completed_runs" "$total_runs" \
                    "E1 $domain scene=$id tau=$threshold"
            done
        done
    done
done
