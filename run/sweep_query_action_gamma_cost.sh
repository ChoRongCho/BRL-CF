#!/usr/bin/env bash
set -uo pipefail

# Internal hyperparameter sweep for the Query-as-Action POMCP baseline.
# Each gamma/query-cost pair is evaluated with the same domain/scene/seed set.

GAMMAS=(0.2 0.5 0.8 0.95)
QUERY_COSTS=(0.0 0.5 1.0)
DOMAINS=(tomato wastesorting)
SCENES=(1 2 3 4 5)

ITERATIONS=10
BASE_SEED=20260921
MAX_STEP="${MAX_STEP:-50}"
N_SIMULATIONS="${N_SIMULATIONS:-100}"
MAX_DEPTH="${MAX_DEPTH:-20}"
FAILURE_PENALTY="${FAILURE_PENALTY:-10.0}"
ANSWER_ACCURACY="${ANSWER_ACCURACY:-1.0}"
DRY_RUN=false
RESUME=false
RUN_ROOT=""

usage() {
    cat <<'EOF'
Usage: ./run/sweep_query_action_gamma_cost.sh [options]

Options:
  --iter N              repetitions per domain/scene (default: 5)
  --base-seed N         first deterministic seed (default: 20260921)
  --run-root PATH       output directory
  --resume              skip episodes already marked complete
  --dry-run             print commands without running experiments
  -h, --help            show this message

Environment overrides:
  N_SIMULATIONS, MAX_DEPTH, MAX_STEP, FAILURE_PENALTY, ANSWER_ACCURACY
EOF
}

while (($#)); do
    case "$1" in
        --iter|--iterations)
            ITERATIONS="${2:?Missing iteration count}"
            shift 2
            ;;
        --base-seed)
            BASE_SEED="${2:?Missing base seed}"
            shift 2
            ;;
        --run-root)
            RUN_ROOT="${2:?Missing run-root path}"
            shift 2
            ;;
        --resume)
            RESUME=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

[[ "$ITERATIONS" =~ ^[1-9][0-9]*$ ]] || {
    echo "--iter must be a positive integer: $ITERATIONS" >&2
    exit 1
}
[[ "$BASE_SEED" =~ ^[0-9]+$ ]] || {
    echo "--base-seed must be a non-negative integer: $BASE_SEED" >&2
    exit 1
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BASELINE_RUN="$PROJECT_ROOT/scripts/baseline/targeted_query_pomdp/run.sh"

if [[ -z "$RUN_ROOT" ]]; then
    RUN_ROOT="$PROJECT_ROOT/experiments_logs/query_action_tuning/$(date +%Y%m%d_%H%M%S)_$$"
elif [[ "$RUN_ROOT" != /* ]]; then
    RUN_ROOT="$PROJECT_ROOT/$RUN_ROOT"
fi

if [[ "$RESUME" == true && ! -d "$RUN_ROOT" ]]; then
    echo "Resume root does not exist: $RUN_ROOT" >&2
    exit 1
fi
if [[ "$RESUME" == false && -e "$RUN_ROOT" ]]; then
    echo "Run root already exists: $RUN_ROOT" >&2
    exit 1
fi

if [[ "$DRY_RUN" == false ]]; then
    mkdir -p "$RUN_ROOT"
fi

manifest="$RUN_ROOT/runs.csv"
if [[ "$DRY_RUN" == false && ! -f "$manifest" ]]; then
    echo "index,gamma,query_cost,domain,scene,iteration,seed,status,run_dir" > "$manifest"
fi

total=$((${#GAMMAS[@]} * ${#QUERY_COSTS[@]} * ${#DOMAINS[@]} * ${#SCENES[@]} * ITERATIONS))
current=0
completed=0
skipped=0
failed=0

label() {
    local value="$1"
    printf '%s' "${value//./p}"
}

echo "Gamma: ${GAMMAS[*]}"
echo "Query cost: ${QUERY_COSTS[*]}"
echo "Domains: ${DOMAINS[*]}; scenes: ${SCENES[*]}; repetitions: $ITERATIONS"
echo "Failure penalty: $FAILURE_PENALTY"
echo "Total episodes: $total"
echo "Run root: $RUN_ROOT"

for gamma in "${GAMMAS[@]}"; do
    gamma_label="$(label "$gamma")"
    for query_cost in "${QUERY_COSTS[@]}"; do
        cost_label="$(label "$query_cost")"
        for domain_index in "${!DOMAINS[@]}"; do
            domain="${DOMAINS[$domain_index]}"
            domain_seed_offset=$((domain_index * 1000000))
            for scene in "${SCENES[@]}"; do
                scene_id=$(printf '%02d' "$((10#$scene))")
                for ((iteration=1; iteration<=ITERATIONS; iteration++)); do
                    current=$((current + 1))
                    # The seed depends only on domain/scene/repetition, so every
                    # hyperparameter pair receives the same stochastic cases.
                    seed=$(((BASE_SEED + domain_seed_offset + 10#$scene * 1000 + iteration) % 4294967296))
                    run_dir="$RUN_ROOT/gamma_${gamma_label}/query_cost_${cost_label}/${domain}/scene_${scene_id}/run_${iteration}"
                    marker="$run_dir/.complete"

                    if [[ "$RESUME" == true && -f "$marker" ]]; then
                        skipped=$((skipped + 1))
                        printf '\rProgress: %d/%d (completed=%d skipped=%d failed=%d)' \
                            "$current" "$total" "$completed" "$skipped" "$failed"
                        continue
                    fi

                    if [[ "$DRY_RUN" == true ]]; then
                        printf '\n[%d/%d] gamma=%s query_cost=%s domain=%s scene=%s iteration=%s seed=%s\n' \
                            "$current" "$total" "$gamma" "$query_cost" "$domain" "$scene" "$iteration" "$seed"
                        printf '  LOG_ROOT=%q DOMAIN=%q SCENE=%q SEED=%q GAMMA=%q QUERY_COST=%q %q\n' \
                            "$run_dir" "$domain" "$scene" "$seed" "$gamma" "$query_cost" "$BASELINE_RUN"
                        continue
                    fi

                    mkdir -p "$run_dir"
                    if LOG_ROOT="$run_dir" \
                        DOMAIN="$domain" \
                        SCENE="$scene" \
                        SEED="$seed" \
                        MAX_STEP="$MAX_STEP" \
                        N_SIMULATIONS="$N_SIMULATIONS" \
                        MAX_DEPTH="$MAX_DEPTH" \
                        GAMMA="$gamma" \
                        QUERY_COST="$query_cost" \
                        FAILURE_PENALTY="$FAILURE_PENALTY" \
                        ANSWER_ACCURACY="$ANSWER_ACCURACY" \
                        "$BASELINE_RUN" > "$run_dir/console.log" 2>&1; then
                        touch "$marker"
                        status=complete
                        completed=$((completed + 1))
                    else
                        status=failed
                        failed=$((failed + 1))
                    fi

                    relative_run_dir="${run_dir#"$RUN_ROOT"/}"
                    echo "$current,$gamma,$query_cost,$domain,$scene,$iteration,$seed,$status,$relative_run_dir" >> "$manifest"
                    printf '\rProgress: %d/%d (completed=%d skipped=%d failed=%d)' \
                        "$current" "$total" "$completed" "$skipped" "$failed"
                done
            done
        done
    done
done

echo
if [[ "$DRY_RUN" == true ]]; then
    echo "Dry run complete. No files were written."
else
    echo "Completed: $completed, skipped: $skipped, failed: $failed"
    echo "Manifest: $manifest"
fi

((failed == 0))
