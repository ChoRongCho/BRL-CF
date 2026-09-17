#!/usr/bin/env bash
# Re-run only Ours for the existing paired query-baseline comparison.
# KnowNo, IntroPlan, and Query-as-Action logs are never touched.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
SEED_LOG="${SEED_LOG:-experiments_logs/system_log/when_what_seed_logs/iterate_when_what_20260912_150132.csv}"
LOG_ROOT="${LOG_ROOT:-experiments_logs/system_log}"
N_SIMULATIONS="${N_SIMULATIONS:-100}"
GAMMA="${GAMMA:-0.2}"
MAX_STEP="${MAX_STEP:-50}"
DRY_RUN="${DRY_RUN:-0}"

[[ -f "$SEED_LOG" ]] || { echo "Seed log not found: $SEED_LOG" >&2; exit 1; }
[[ "$N_SIMULATIONS" =~ ^[1-9][0-9]*$ ]] || { echo "N_SIMULATIONS must be positive" >&2; exit 1; }

mapfile -t runs < <(awk -F, 'NR>1 && $2=="ours" {print $1","$3","$4","$5","$6","$7}' "$SEED_LOG")
[[ ${#runs[@]} -eq 400 ]] || { echo "Expected 400 paired Ours rows, found ${#runs[@]}" >&2; exit 1; }

timestamp=$(date +%Y%m%d_%H%M%S)
if [[ "$DRY_RUN" != 1 ]]; then
    for domain in tomato wastesorting; do
        for scene in 1 2 3 4 5; do
            printf -v scene_id '%02d' "$scene"
            target="$LOG_ROOT/$domain/scene_${scene_id}_step${MAX_STEP}/when_what_ours_thres_0-8_rand_0-4"
            [[ ! -d "$target" ]] || mv "$target" "${target}.backup_${timestamp}"
        done
    done
fi

current=0
for row in "${runs[@]}"; do
    IFS=, read -r domain scene iteration seed threshold probability <<< "$row"
    current=$((current + 1))
    command=(bash run/run_when_what_ablation.sh --domain "$domain" --scene "$scene" --iter 1
        --condition ours --threshold "$threshold" --random-query-prob "$probability"
        --max-step "$MAX_STEP" --seed "$seed" --log-root "$LOG_ROOT"
        --n-simulations "$N_SIMULATIONS" --gamma "$GAMMA")
    printf '\rOurs: %3d%% (%d/400)' "$((current * 100 / 400))" "$current"
    if [[ "$DRY_RUN" == 1 ]]; then
        printf '\n'; printf '%q ' "${command[@]}"; printf '\n'
    else
        "${command[@]}" >/dev/null
    fi
done
printf '\n'

if [[ "$DRY_RUN" != 1 ]]; then
    python3 experiments/system_eval/analysis_experiment.py
    python3 experiments/system_eval/plot_query_baseline_figure.py
    echo "Updated Ours only: n_simulations=$N_SIMULATIONS gamma=$GAMMA"
fi
