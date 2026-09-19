#!/usr/bin/env bash
# Run one or all domain/method combinations after a large refactor.
# Default is a dry run. Add --execute only after reviewing the commands.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

domain_selection="all"
method_selection="all"
scene="01"
seed="42"
execute="false"

usage() {
    cat <<'EOF'
Usage: bash run/check_refactor_matrix.sh [options]

Options:
  --domain tomato|wastesorting|all
  --method ours|knowno|introplan|query_action_pomdp|all
  --scene N
  --seed N
  --execute    Run the selected commands. Without this flag, only print them.

Examples:
  bash run/check_refactor_matrix.sh --domain tomato --method ours
  bash run/check_refactor_matrix.sh --domain tomato --method ours --execute
  bash run/check_refactor_matrix.sh --execute
EOF
}

while (($#)); do
    case "$1" in
        --domain) domain_selection="${2:?Missing domain}"; shift 2 ;;
        --method) method_selection="${2:?Missing method}"; shift 2 ;;
        --scene) scene="${2:?Missing scene}"; shift 2 ;;
        --seed) seed="${2:?Missing seed}"; shift 2 ;;
        --execute) execute="true"; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
    esac
done

case "$domain_selection" in
    tomato|wastesorting) domains=("$domain_selection") ;;
    all) domains=(tomato wastesorting) ;;
    *) echo "Unknown domain: $domain_selection" >&2; exit 1 ;;
esac

case "$method_selection" in
    ours|knowno|introplan|query_action_pomdp) methods=("$method_selection") ;;
    all) methods=(ours knowno introplan query_action_pomdp) ;;
    *) echo "Unknown method: $method_selection" >&2; exit 1 ;;
esac

[[ "$scene" =~ ^[1-9][0-9]*$|^0[1-9][0-9]*$ ]] || {
    echo "SCENE must be a positive integer: $scene" >&2
    exit 1
}
[[ "$seed" =~ ^(0|[1-9][0-9]{0,9})$ ]] && ((seed <= 4294967295)) || {
    echo "SEED must be in [0, 4294967295]: $seed" >&2
    exit 1
}
printf -v scene_id '%02d' "$((10#$scene))"

for domain in "${domains[@]}"; do
    for filename in "scene_${scene_id}.yaml" domain_rule.yaml robot_skill.yaml env_setting.yaml; do
        [[ -f "scripts/domain/$domain/$filename" ]] || {
            echo "Missing scripts/domain/$domain/$filename" >&2
            exit 1
        }
    done
done

python="${PYTHON:-python3}"
python_path="$($python -c 'import sys; print(sys.executable)')"
# The baseline wrappers invoke `python3` internally. Put the selected
# interpreter first so every method runs in the same environment as Ours.
export PATH="$(dirname "$python_path"):$PATH"
n_simulations="${N_SIMULATIONS:-100}"
ours_gamma="${OURS_GAMMA:-0.2}"
query_action_gamma="${QUERY_ACTION_GAMMA:-0.95}"
max_step="${MAX_STEP:-50}"
stamp="$(date +%Y%m%d_%H%M%S)_$$"
log_root="${LOG_ROOT:-experiments_logs/refactor_smoke/$stamp}"

run_one() {
    local domain="$1"
    local method="$2"
    local env_setting="scripts/domain/$domain/env_setting.yaml"
    local combo_root="$log_root/$method/$domain/scene_$scene_id"
    local combo_root_abs
    local -a command

    if [[ "$combo_root" == /* ]]; then
        combo_root_abs="$combo_root"
    else
        combo_root_abs="$PWD/$combo_root"
    fi

    case "$method" in
        ours)
            command=("$python" main.py
                --domain "$domain"
                --initial_state "scripts/domain/$domain/scene_${scene_id}.yaml"
                --domain_rule "scripts/domain/$domain/domain_rule.yaml"
                --robot_skill "scripts/domain/$domain/robot_skill.yaml"
                --env-setting "$env_setting"
                --n_simulations "$n_simulations"
                --gamma "$ours_gamma"
                --max_step "$max_step"
                --seed "$seed"
                --answer_type auto
                --log_dir "$combo_root")
            ;;
        knowno|introplan)
            command=(env
                "BASELINE=$method"
                "DOMAIN=$domain"
                "SCENE=$scene_id"
                "SEED=$seed"
                "MAX_STEPS=$max_step"
                "ENV_SETTING=$PWD/$env_setting"
                "LOG_FILE=$combo_root_abs/run.txt"
                "AUTO_ANSWER=true"
                "DRY_RUN=false"
                bash run/run_query_baseline.sh)
            ;;
        query_action_pomdp)
            command=(env
                "BASELINE=query_action_pomdp"
                "DOMAIN=$domain"
                "SCENE=$scene_id"
                "SEED=$seed"
                "MAX_STEP=$max_step"
                "N_SIMULATIONS=$n_simulations"
                "GAMMA=$query_action_gamma"
                "ENV_SETTING=$PWD/$env_setting"
                "LOG_ROOT=$combo_root_abs"
                "DRY_RUN=false"
                bash run/run_query_baseline.sh)
            ;;
    esac

    printf '[%s/%s] ' "$domain" "$method"
    printf '%q ' "${command[@]}"
    printf '\n'
    if [[ "$execute" != "true" ]]; then
        return
    fi

    mkdir -p "$combo_root"
    printf '%q ' "${command[@]}" > "$combo_root/command.txt"
    printf '\n' >> "$combo_root/command.txt"
    "${command[@]}" 2>&1 | tee "$combo_root/console.log"
}

for domain in "${domains[@]}"; do
    for method in "${methods[@]}"; do
        run_one "$domain" "$method"
    done
done

if [[ "$execute" == "true" ]]; then
    echo "Smoke-test logs: $log_root"
else
    echo "Dry run only. Add --execute to run the selected combination(s)."
fi
