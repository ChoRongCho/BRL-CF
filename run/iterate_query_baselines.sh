#!/usr/bin/env bash

set -euo pipefail

# ==================== Shared experiment configuration ====================
BASELINES=(knowno query_action_pomcp)
read -r -a DOMAINS <<< "${BATCH_DOMAINS:-tomato wastesorting}"
read -r -a SCENES <<< "${BATCH_SCENES:-1 2 3 4 5}"
ITERATIONS="${BATCH_ITERATIONS:-40}"
MAX_STEPS="${BATCH_MAX_STEPS:-50}"
PAIRED_SEED_LOG="${BATCH_PAIRED_SEED_LOG:-experiments_logs/system_log/when_what_seed_logs/iterate_when_what_20260912_150132.csv}"
LOG_ROOT="${BATCH_LOG_ROOT:-experiments_logs/system_log}"

# KnowNo
# The historical 02 baseline results were produced with the longer, state-aware
# v2 prompt.  Keep the batch runner on that prompt so results are comparable.
PROMPT_VERSION="${BATCH_PROMPT_VERSION:-v2}"
SCORE_TEMPERATURE="${BATCH_SCORE_TEMPERATURE:-5.0}"
TOMATO_QHAT="${BATCH_TOMATO_QHAT:-0.8404}"
WASTE_QHAT="${BATCH_WASTE_QHAT:-0.8704}"

# Query-Action POMCP
N_SIMULATIONS="100"
MAX_DEPTH="20"
GAMMA="0.95"
UCB_C="1.0"
EPSILON="0.005"
MAX_PARTICLES="250"
MAX_BELIEF_PARTICLES="8000"
MAX_NODE_PARTICLES="8000"
QUERY_COST="1.0"
FAILURE_PENALTY="${BATCH_FAILURE_PENALTY:-10.0}"
ANSWER_ACCURACY="1.0"
MAX_CONSECUTIVE_QUERIES="30"

ARCHIVE_EXISTING="${BATCH_ARCHIVE_EXISTING:-true}"
RESUME="${BATCH_RESUME:-false}"
DRY_RUN="${BATCH_DRY_RUN:-false}"
# ========================================================================

usage() {
    echo "Usage: $0 [--dry-run] [--resume] [--no-archive] [--iter N] [--baseline NAME]"
    echo "Runs KnowNo and Query-Action POMCP with the same paired Ours seeds."
}

while (($#)); do
    case "$1" in
        --dry-run) DRY_RUN="true"; shift ;;
        --resume) RESUME="true"; ARCHIVE_EXISTING="false"; shift ;;
        --no-archive) ARCHIVE_EXISTING="false"; shift ;;
        --iter|--iteration) ITERATIONS="${2:?Missing value for $1}"; shift 2 ;;
        --baseline)
            case "${2:?Missing value for $1}" in
                knowno) BASELINES=(knowno) ;;
                query_action_pomcp|query_action_pomdp|query-as-action|targeted_query_pomdp)
                    BASELINES=(query_action_pomcp) ;;
                *) echo "Unsupported baseline: $2" >&2; exit 1 ;;
            esac
            shift 2
            ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
    esac
done

if ! [[ "$ITERATIONS" =~ ^[1-9][0-9]*$ ]]; then
    echo "iterations must be a positive integer." >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
timestamp="$(date +%Y%m%d_%H%M%S)"
total=$((${#BASELINES[@]} * ${#DOMAINS[@]} * ${#SCENES[@]} * ITERATIONS))
current=0
failed=0
skipped=0

cd "$PROJECT_ROOT"
if [[ ! -f "$PAIRED_SEED_LOG" ]]; then
    echo "Paired seed log not found: $PAIRED_SEED_LOG" >&2
    exit 1
fi

if [[ "$ARCHIVE_EXISTING" == "true" && "$DRY_RUN" != "true" ]]; then
    archive_root="${LOG_ROOT}/archive/query_baselines_${timestamp}"
    archive_names=()
    for baseline in "${BASELINES[@]}"; do
        if [[ "$baseline" == "knowno" ]]; then
            archive_names+=(when_knowno_gpt4)
        else
            archive_names+=(query_as_action)
        fi
    done
    for domain in "${DOMAINS[@]}"; do
        for scene in "${SCENES[@]}"; do
            scene_id=$(printf "%02d" "$((10#$scene))")
            scene_root="${LOG_ROOT}/${domain}/scene_${scene_id}_step${MAX_STEPS}"
            for log_name in "${archive_names[@]}"; do
                target="${scene_root}/${log_name}"
                if [[ -d "$target" ]]; then
                    destination="${archive_root}/${domain}/scene_${scene_id}_step${MAX_STEPS}"
                    mkdir -p "$destination"
                    mv "$target" "$destination/$log_name"
                fi
            done
        done
    done
fi

seed_dir="${LOG_ROOT}/query_baseline_seed_logs"
seed_log="${seed_dir}/iterate_query_baselines_${timestamp}.csv"
if [[ "$DRY_RUN" != "true" ]]; then
    mkdir -p "$seed_dir"
    echo "global_index,baseline,domain,scene,iteration,seed,max_steps,n_simulations,max_depth,query_cost,failure_penalty,answer_accuracy,prompt_version,qhat,temperature,expert,status,log_path" > "$seed_log"
fi

printf "\rProgress: %3d%%" 0
for baseline in "${BASELINES[@]}"; do
    while IFS=, read -r domain condition scene iteration seed threshold random_query_prob; do
        if [[ "$domain" == "domain" || "$condition" != "ours" || "$iteration" -gt "$ITERATIONS" ]]; then
            continue
        fi

        selected="false"
        for configured_domain in "${DOMAINS[@]}"; do
            for configured_scene in "${SCENES[@]}"; do
                if [[ "$domain" == "$configured_domain" && "$scene" == "$configured_scene" ]]; then
                    selected="true"
                fi
            done
        done
        [[ "$selected" == "true" ]] || continue

        current=$((current + 1))
        scene_id=$(printf "%02d" "$((10#$scene))")
        qhat=""
        [[ "$domain" == "tomato" ]] && qhat="$TOMATO_QHAT"
        [[ "$domain" == "wastesorting" ]] && qhat="$WASTE_QHAT"

        if [[ "$baseline" == "knowno" ]]; then
            log_dir="${LOG_ROOT}/${domain}/scene_${scene_id}_step${MAX_STEPS}/when_knowno_gpt4"
            log_path="${log_dir}/knowno_${domain}_scene${scene_id}_iter$(printf '%02d' "$iteration")_seed${seed}.txt"
        else
            log_dir="${LOG_ROOT}/${domain}/scene_${scene_id}_step${MAX_STEPS}/query_as_action"
            log_path="$log_dir"
        fi
        done_dir="${log_dir}/.completed"
        done_file="${done_dir}/${baseline}_${domain}_scene${scene_id}_iter$(printf '%02d' "$iteration")_seed${seed}.done"

        if [[ "$RESUME" == "true" && -f "$done_file" ]]; then
            skipped=$((skipped + 1))
            printf "\rProgress: %3d%%" "$((current * 100 / total))"
            continue
        fi

        if [[ "$DRY_RUN" == "true" ]]; then
            # Print one representative command per baseline while still
            # validating every paired-seed row.
            if [[ "$domain" == "tomato" && "$scene" == "1" && "$iteration" == "1" ]]; then
                BASELINE="$baseline" DOMAIN="$domain" SCENE="$scene_id" SEED="$seed" \
                MAX_STEPS="$MAX_STEPS" MAX_STEP="$MAX_STEPS" QHAT="$qhat" \
                PROMPT_VERSION="$PROMPT_VERSION" SCORE_TEMPERATURE="$SCORE_TEMPERATURE" \
                N_SIMULATIONS="$N_SIMULATIONS" MAX_DEPTH="$MAX_DEPTH" \
                GAMMA="$GAMMA" UCB_C="$UCB_C" EPSILON="$EPSILON" \
                MAX_PARTICLES="$MAX_PARTICLES" \
                MAX_BELIEF_PARTICLES="$MAX_BELIEF_PARTICLES" \
                MAX_NODE_PARTICLES="$MAX_NODE_PARTICLES" \
                QUERY_COST="$QUERY_COST" ANSWER_ACCURACY="$ANSWER_ACCURACY" \
                FAILURE_PENALTY="$FAILURE_PENALTY" \
                MAX_CONSECUTIVE_QUERIES="$MAX_CONSECUTIVE_QUERIES" \
                LOG_ROOT="$LOG_ROOT" LOG_FILE="$log_path" AUTO_ANSWER="true" DRY_RUN="true" \
                    "$SCRIPT_DIR/run_query_baseline.sh"
            fi
        else
            mkdir -p "$log_dir" "$done_dir"
            status="complete"
            if BASELINE="$baseline" DOMAIN="$domain" SCENE="$scene_id" SEED="$seed" \
                MAX_STEPS="$MAX_STEPS" MAX_STEP="$MAX_STEPS" QHAT="$qhat" \
                PROMPT_VERSION="$PROMPT_VERSION" SCORE_TEMPERATURE="$SCORE_TEMPERATURE" \
                N_SIMULATIONS="$N_SIMULATIONS" MAX_DEPTH="$MAX_DEPTH" \
                GAMMA="$GAMMA" UCB_C="$UCB_C" EPSILON="$EPSILON" \
                MAX_PARTICLES="$MAX_PARTICLES" \
                MAX_BELIEF_PARTICLES="$MAX_BELIEF_PARTICLES" \
                MAX_NODE_PARTICLES="$MAX_NODE_PARTICLES" \
                QUERY_COST="$QUERY_COST" ANSWER_ACCURACY="$ANSWER_ACCURACY" \
                FAILURE_PENALTY="$FAILURE_PENALTY" \
                MAX_CONSECUTIVE_QUERIES="$MAX_CONSECUTIVE_QUERIES" \
                LOG_ROOT="$LOG_ROOT" LOG_FILE="$log_path" AUTO_ANSWER="true" \
                    "$SCRIPT_DIR/run_query_baseline.sh" >/dev/null; then
                touch "$done_file"
            else
                status="failed"
                failed=$((failed + 1))
            fi
            echo "${current},${baseline},${domain},${scene_id},${iteration},${seed},${MAX_STEPS},${N_SIMULATIONS},${MAX_DEPTH},${QUERY_COST},${FAILURE_PENALTY},${ANSWER_ACCURACY},${PROMPT_VERSION},${qhat},${SCORE_TEMPERATURE},exact_oracle,${status},${log_path}" >> "$seed_log"
        fi
        printf "\rProgress: %3d%%" "$((current * 100 / total))"
    done < "$PAIRED_SEED_LOG"
done

if ((current != total)); then
    echo
    echo "Expected $total paired runs but found $current." >&2
    exit 1
fi

printf "\rProgress: 100%%\n"
if [[ "$DRY_RUN" == "true" ]]; then
    echo "Dry run checked $current paired runs; no API calls or experiments were executed."
else
    echo "Seed log saved to $seed_log"
    echo "Completed: $((current - failed - skipped)), skipped: $skipped, failed: $failed"
fi
