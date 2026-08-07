#!/usr/bin/env bash
set -euo pipefail

# E4: Feedback-source study: ours/knowno x vlm/human.
# Oracle references come from E1-E3 and are not rerun here.
source "$(dirname "$0")/common.sh"

# E4 global settings. Environment variables override these defaults.
E4_ITERATIONS="${E4_ITERATIONS:-40}"
E4_DOMAINS="${E4_DOMAINS:-tomato wastesorting}"
E4_SCENES="${E4_SCENES:-1 2 3 4 5}"
E4_METHODS="${E4_METHODS:-ours knowno}"
E4_PROVIDERS="${E4_PROVIDERS:-vlm human}"

total_runs=$(($(word_count "$E4_METHODS") * $(word_count "$E4_PROVIDERS") * $(word_count "$E4_DOMAINS") * $(word_count "$E4_SCENES") * E4_ITERATIONS))
completed_runs=0
progress_update "$completed_runs" "$total_runs" "E4 feedback sources"

cd "$ROOT"
for provider in $E4_PROVIDERS; do
    case "$provider" in
        vlm)
            echo "PLACEHOLDER: GPT VLM provider is not implemented; skipping *-vlm conditions." >&2
            skipped_runs=$(($(word_count "$E4_METHODS") * $(word_count "$E4_DOMAINS") * $(word_count "$E4_SCENES") * E4_ITERATIONS))
            completed_runs=$((completed_runs + skipped_runs))
            progress_update "$completed_runs" "$total_runs" "E4 vlm skipped"
            continue
            ;;
        human) ;;
        *) echo "E4_PROVIDERS may contain only: vlm human" >&2; exit 1 ;;
    esac

    for method in $E4_METHODS; do
        for domain in $E4_DOMAINS; do
            for scene in $E4_SCENES; do
                id="$(scene_id "$scene")"
                log_dir="$(experiment_log_dir e4_feedback "$domain" "$scene" "$method" "$provider")"
                for ((run_index = 1; run_index <= E4_ITERATIONS; run_index++)); do
                    run_seed="$(experiment_seed "$domain" "$scene" "$run_index")"
                    case "$method" in
                        ours)
                            run_ours "$domain" "$scene" "$TAU_GLOBAL" human "$run_seed" "$log_dir"
                            ;;
                        knowno)
                            run_knowno "$domain" "$scene" human "$run_seed" \
                                "$log_dir/run_${run_index}.json"
                            ;;
                        *) echo "Unknown E4 method: $method" >&2; exit 1 ;;
                    esac
                    completed_runs=$((completed_runs + 1))
                    progress_update "$completed_runs" "$total_runs" \
                        "E4 $method $provider $domain scene=$id"
                done
            done
        done
    done
done
