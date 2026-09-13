#!/usr/bin/env bash

set -euo pipefail

# Experiment globals. Override any value from the shell, e.g.
# QUERY_COST=1.0 N_SIMULATIONS=100 ./scripts/baseline/targeted_query_pomdp/run.sh
DOMAIN="${DOMAIN:-tomato}"
SCENE="${SCENE:-01}"
SEED="${SEED:-random}"
MAX_STEP="${MAX_STEP:-50}"
N_SIMULATIONS="${N_SIMULATIONS:-100}"
MAX_DEPTH="${MAX_DEPTH:-20}"
QUERY_COST="${QUERY_COST:-1.0}"
ANSWER_ACCURACY="${ANSWER_ACCURACY:-1.0}"
MAX_CONSECUTIVE_QUERIES="${MAX_CONSECUTIVE_QUERIES:-30}"
LOG_ROOT="${LOG_ROOT:-experiments_logs/system_log}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
scene_id=$(printf "%02d" "$((10#$SCENE))")

if [[ "$SEED" == "random" ]]; then
    SEED=$(od -An -N4 -tu4 /dev/urandom | tr -d ' ')
fi

initial_state="scripts/domain/${DOMAIN}/scene_${scene_id}.yaml"
domain_rule="scripts/domain/${DOMAIN}/domain_rule.yaml"
robot_skill="scripts/domain/${DOMAIN}/robot_skill.yaml"
log_dir="${LOG_ROOT}/${DOMAIN}/scene_${scene_id}_step${MAX_STEP}/query_as_action"

cd "$PROJECT_ROOT"
python3 scripts/baseline/targeted_query_pomdp/run_experiment.py \
    --domain "$DOMAIN" \
    --domain_rule "$domain_rule" \
    --initial_state "$initial_state" \
    --robot_skill "$robot_skill" \
    --answer_type auto \
    --seed "$SEED" \
    --max_step "$MAX_STEP" \
    --n_simulations "$N_SIMULATIONS" \
    --max_depth "$MAX_DEPTH" \
    --log_dir "$log_dir" \
    --query-cost "$QUERY_COST" \
    --answer-accuracy "$ANSWER_ACCURACY" \
    --max-consecutive-queries "$MAX_CONSECUTIVE_QUERIES"
