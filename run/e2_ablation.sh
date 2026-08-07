#!/usr/bin/env bash
set -euo pipefail

# E2: Oracle ablation from 10_TODO. Canonical methods are listed below.
source "$(dirname "$0")/common.sh"

# E2 global settings. Environment variables override these defaults.
E2_ITERATIONS="${E2_ITERATIONS:-40}"
E2_DOMAINS="${E2_DOMAINS:-tomato wastesorting}"
E2_SCENES="${E2_SCENES:-1 2 3 4 5}"
E2_METHODS="${E2_METHODS:-ours ours-random-when ours-random-what}"

total_runs=$(($(word_count "$E2_METHODS") * $(word_count "$E2_DOMAINS") * $(word_count "$E2_SCENES") * E2_ITERATIONS))
completed_runs=0
progress_update "$completed_runs" "$total_runs" "E2 ablation"

cd "$ROOT"
for method in $E2_METHODS; do
    for domain in $E2_DOMAINS; do
        for scene in $E2_SCENES; do
            id="$(scene_id "$scene")"
            for ((run_index = 1; run_index <= E2_ITERATIONS; run_index++)); do
                run_seed="$(experiment_seed "$domain" "$scene" "$run_index")"
                log_dir="$(experiment_log_dir e2_ablation "$domain" "$scene" "$method" oracle)"
                case "$method" in
                    ours)
                        run_ours "$domain" "$scene" "$TAU_GLOBAL" oracle "$run_seed" "$log_dir"
                        ;;
                    ours-random-when)
                        run_random_when "$domain" "$scene" "$run_seed" "$log_dir"
                        ;;
                    ours-random-what)
                        run_random_what "$domain" "$scene" "$run_seed" "$log_dir"
                        ;;
                    *) echo "Unknown E2 method: $method" >&2; exit 1 ;;
                esac
                completed_runs=$((completed_runs + 1))
                progress_update "$completed_runs" "$total_runs" \
                    "E2 $method $domain scene=$id"
            done
        done
    done
done
