#!/usr/bin/env bash
# Each simulation budget: 5 scenes x 10 repetitions per domain.
# Example: N_SIMULATIONS="100 500" DOMAINS="tomato" bash run/iterate_n_simulations.sh
# DRY_RUN=1 prints commands without running experiments or creating logs.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
read -r -a budgets <<< "${N_SIMULATIONS:-50 100 200}"
read -r -a domains <<< "${DOMAINS:-tomato wastesorting}"
read -r -a scenes <<< "${SCENES:-1 2 3 4 5}"
iterations="${ITERATIONS:-10}"
base_seed="${SEED:-42}"
python="${PYTHON:-python3}"
dry_run="${DRY_RUN:-0}"
log_root="${LOG_ROOT:-experiments_logs/n_simulations/$(date +%Y%m%d_%H%M%S)_$$}"

for number in "$iterations" "${budgets[@]}" "${scenes[@]}"; do
    [[ "$number" =~ ^[1-9][0-9]*$ ]] || { echo "Expected positive integer: $number" >&2; exit 1; }
done
[[ "$base_seed" =~ ^(0|[1-9][0-9]{0,9})$ ]] && ((base_seed <= 4294967295)) || {
    echo "SEED must be an integer in [0, 4294967295]" >&2; exit 1;
}
for domain in "${domains[@]}"; do
    [[ "$domain" == tomato || "$domain" == wastesorting ]] || { echo "Unknown domain: $domain" >&2; exit 1; }
    for scene in "${scenes[@]}"; do
        printf -v scene_id '%02d' "$scene"
        for file in "scene_${scene_id}.yaml" domain_rule.yaml robot_skill.yaml; do
            [[ -f "scripts/domain/$domain/$file" ]] || { echo "Missing $domain/$file" >&2; exit 1; }
        done
    done
done

if [[ "$dry_run" != 1 ]]; then
    # Refuse reuse so previous results cannot silently contaminate this sweep.
    [[ ! -e "$log_root" ]] || { echo "LOG_ROOT already exists: $log_root" >&2; exit 1; }
    mkdir -p "$log_root"
    printf 'domain,scene,iteration,n_simulations,seed,exit_code,run_dir\n' > "$log_root/runs.csv"
fi
total=$((${#budgets[@]} * ${#domains[@]} * ${#scenes[@]} * iterations))
current=0
failures=0
for domain in "${domains[@]}"; do
    domain_offset=0
    [[ "$domain" != wastesorting ]] || domain_offset=1000000
    for scene in "${scenes[@]}"; do
        printf -v scene_id '%02d' "$scene"
        for ((iteration=1; iteration<=iterations; iteration++)); do
            seed=$(((base_seed + domain_offset + scene * iterations + iteration) % 4294967296))
            for budget in "${budgets[@]}"; do
                current=$((current + 1))
                run_dir="$log_root/$domain/sim_$budget/scene_$scene_id/run_$iteration"
                command=("$python" main.py
                    --domain "$domain"
                    --initial_state "scripts/domain/$domain/scene_$scene_id.yaml"
                    --domain_rule "scripts/domain/$domain/domain_rule.yaml"
                    --robot_skill "scripts/domain/$domain/robot_skill.yaml"
                    --n_simulations "$budget" --seed "$seed" --log_dir "$run_dir"
                    --gamma "${GAMMA:-0.2}" --c "${C:-1.0}"
                    --max_depth "${MAX_DEPTH:-20}" --epsilon "${EPSILON:-0.005}"
                    --threshold "${THRESHOLD:-0.8}" --max_step "${MAX_STEP:-50}"
                    --answer_type auto)
                printf '[%s/%s] %s scene=%s repeat=%s simulations=%s seed=%s\n' \
                    "$current" "$total" "$domain" "$scene_id" "$iteration" "$budget" "$seed"
                if [[ "$dry_run" == 1 ]]; then
                    printf '%q ' "${command[@]}"; printf '\n'
                    continue
                fi
                mkdir -p "$run_dir"
                printf '%q ' "${command[@]}" > "$run_dir/command.txt"
                status=0
                "${command[@]}" > "$run_dir/console.log" 2>&1 || status=$?
                printf '%s,%s,%s,%s,%s,%s,%s\n' "$domain" "$scene_id" "$iteration" \
                    "$budget" "$seed" "$status" "$domain/sim_$budget/scene_$scene_id/run_$iteration" >> "$log_root/runs.csv"
                if ((status != 0)); then
                    failures=$((failures + 1))
                    echo "Execution error ($status): $run_dir/console.log" >&2
                fi
            done
        done
    done
done
if [[ "$dry_run" != 1 ]]; then
    "$python" experiments/system_eval/summarize_n_simulations.py "$log_root"
    echo "Results: $log_root ($failures execution errors)"
    ((failures == 0))
fi
