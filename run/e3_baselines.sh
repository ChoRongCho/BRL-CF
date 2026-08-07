#!/usr/bin/env bash
set -euo pipefail

# E3: Oracle comparison from 10_TODO: ours, knowno, active-search.
source "$(dirname "$0")/common.sh"

# E3 global settings. Environment variables override these defaults.
E3_ITERATIONS="${E3_ITERATIONS:-40}"
E3_DOMAINS="${E3_DOMAINS:-tomato wastesorting}"
E3_SCENES="${E3_SCENES:-1 2 3 4 5}"
E3_METHODS="${E3_METHODS:-ours knowno active-search}"

total_runs=$(($(word_count "$E3_METHODS") * $(word_count "$E3_DOMAINS") * $(word_count "$E3_SCENES") * E3_ITERATIONS))
completed_runs=0
progress_update "$completed_runs" "$total_runs" "E3 baselines"

cd "$ROOT"
for method in $E3_METHODS; do
    for domain in $E3_DOMAINS; do
        for scene in $E3_SCENES; do
            id="$(scene_id "$scene")"
            log_dir="$(experiment_log_dir e3_baselines "$domain" "$scene" "$method" oracle)"
            for ((run_index = 1; run_index <= E3_ITERATIONS; run_index++)); do
                run_seed="$(experiment_seed "$domain" "$scene" "$run_index")"
                mkdir -p "$log_dir"
                case "$method" in
                    ours)
                        run_ours "$domain" "$scene" "$TAU_GLOBAL" oracle "$run_seed" "$log_dir"
                        ;;
                    active-search)
                        # active-search is the grounded-fact Adapted Attr-POMDP
                        # query-action planner defined in 10_TODO.
                        run_quietly "$log_dir/console_seed_${run_seed}.log" \
                            python3 scripts/baseline/attr_pomdp/main.py \
                                --domain "$domain" --scene "$id" --seed "$run_seed" \
                                --max-step "$MAX_STEP" --max-depth 2 --log-dir "$log_dir" \
                                --feedback-source oracle
                        ;;
                    knowno)
                        run_knowno "$domain" "$scene" oracle "$run_seed" \
                            "$log_dir/run_${run_index}.json"
                        ;;
                    *)
                        echo "Unknown E3 method: $method" >&2
                        exit 1
                        ;;
                esac
                completed_runs=$((completed_runs + 1))
                progress_update "$completed_runs" "$total_runs" \
                    "E3 $method $domain scene=$id"
            done
        done
    done
done
