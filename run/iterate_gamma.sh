#!/usr/bin/env bash
# Gamma sweep: 5 scenes x 10 repetitions per domain, n_simulations fixed at 100.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
read -r -a gammas <<< "${GAMMAS:-0.4 0.3 0.2 0.1 0.05 0.0}"
read -r -a domains <<< "${DOMAINS:-tomato wastesorting}"
read -r -a scenes <<< "${SCENES:-1 2 3 4 5}"
iterations="${ITERATIONS:-10}"; simulations="${N_SIMULATIONS:-100}"; base_seed="${SEED:-42}"; python="${PYTHON:-python3}"; dry_run="${DRY_RUN:-0}"
log_root="${LOG_ROOT:-experiments_logs/gamma/$(date +%Y%m%d_%H%M%S)_$$}"
[[ "$iterations" =~ ^[1-9][0-9]*$ && "$simulations" =~ ^[1-9][0-9]*$ ]] || { echo "iterations and n_simulations must be positive integers" >&2; exit 1; }
for gamma in "${gammas[@]}"; do [[ "$gamma" =~ ^0([.][0-9]+)?$|^1([.]0+)?$ ]] || { echo "Gamma must be in [0,1]: $gamma" >&2; exit 1; }; done
for domain in "${domains[@]}"; do
    [[ "$domain" == tomato || "$domain" == wastesorting ]] || { echo "Unknown domain: $domain" >&2; exit 1; }
    for scene in "${scenes[@]}"; do printf -v id '%02d' "$scene"; for file in "scene_${id}.yaml" domain_rule.yaml robot_skill.yaml; do [[ -f "scripts/domain/$domain/$file" ]] || { echo "Missing $domain/$file" >&2; exit 1; }; done; done
done
if [[ "$dry_run" != 1 ]]; then [[ ! -e "$log_root" ]] || { echo "LOG_ROOT already exists: $log_root" >&2; exit 1; }; mkdir -p "$log_root"; printf 'domain,scene,iteration,gamma,n_simulations,seed,exit_code,run_dir\n' > "$log_root/runs.csv"; fi
total=$((${#gammas[@]} * ${#domains[@]} * ${#scenes[@]} * iterations)); current=0; failures=0
for domain in "${domains[@]}"; do offset=0; [[ "$domain" != wastesorting ]] || offset=1000000; for scene in "${scenes[@]}"; do printf -v id '%02d' "$scene"; for ((iteration=1; iteration<=iterations; iteration++)); do seed=$(((base_seed + offset + scene * iterations + iteration) % 4294967296)); for gamma in "${gammas[@]}"; do
    current=$((current+1)); label="${gamma/./-}"; run_dir="$log_root/$domain/gamma_$label/scene_$id/run_$iteration"
    command=("$python" main.py --domain "$domain" --initial_state "scripts/domain/$domain/scene_$id.yaml" --domain_rule "scripts/domain/$domain/domain_rule.yaml" --robot_skill "scripts/domain/$domain/robot_skill.yaml" --n_simulations "$simulations" --gamma "$gamma" --seed "$seed" --log_dir "$run_dir" --c "${C:-1.0}" --max_depth "${MAX_DEPTH:-20}" --epsilon "${EPSILON:-0.005}" --threshold "${THRESHOLD:-0.8}" --max_step "${MAX_STEP:-50}" --answer_type auto)
    printf '[%s/%s] %s scene=%s repeat=%s gamma=%s simulations=%s\n' "$current" "$total" "$domain" "$id" "$iteration" "$gamma" "$simulations"
    if [[ "$dry_run" == 1 ]]; then printf '%q ' "${command[@]}"; printf '\n'; continue; fi
    mkdir -p "$run_dir"; printf '%q ' "${command[@]}" > "$run_dir/command.txt"; status=0; "${command[@]}" > "$run_dir/console.log" 2>&1 || status=$?
    printf '%s,%s,%s,%s,%s,%s,%s,%s\n' "$domain" "$id" "$iteration" "$gamma" "$simulations" "$seed" "$status" "$domain/gamma_$label/scene_$id/run_$iteration" >> "$log_root/runs.csv"
    if ((status != 0)); then failures=$((failures+1)); echo "Execution error ($status): $run_dir/console.log" >&2; fi
done; done; done; done
if [[ "$dry_run" != 1 ]]; then "$python" experiments/system_eval/summarize_gamma.py "$log_root"; echo "Results: $log_root ($failures execution errors)"; ((failures == 0)); fi
