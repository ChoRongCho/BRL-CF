#!/usr/bin/env bash

set -euo pipefail

DOMAIN="${DOMAIN:-tomato}"
SCENE="${SCENE:-01}"
SEED="${SEED:-random}"
MAX_STEP="${MAX_STEP:-50}"
ATTR_DEPTH="${ATTR_DEPTH:-3}"
ATTRIBUTE_COST="${ATTRIBUTE_COST:-0.1}"
ANSWER_ACCURACY="${ANSWER_ACCURACY:-0.99}"
MAX_QUESTIONS_PER_ACTION="${MAX_QUESTIONS_PER_ACTION:-10}"
MAX_CANDIDATE_QUESTIONS="${MAX_CANDIDATE_QUESTIONS:-8}"
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
log_dir="${LOG_ROOT}/${DOMAIN}/scene_${scene_id}_step${MAX_STEP}/attr_pomdp"

cd "$PROJECT_ROOT"
python3 scripts/baseline/attr_pomdp/run_experiment.py \
    --domain "$DOMAIN" \
    --domain_rule "$domain_rule" \
    --initial_state "$initial_state" \
    --robot_skill "$robot_skill" \
    --answer_type auto \
    --seed "$SEED" \
    --max_step "$MAX_STEP" \
    --log_dir "$log_dir" \
    --attr-depth "$ATTR_DEPTH" \
    --attribute-cost "$ATTRIBUTE_COST" \
    --answer-accuracy "$ANSWER_ACCURACY" \
    --max-questions-per-action "$MAX_QUESTIONS_PER_ACTION" \
    --max-candidate-questions "$MAX_CANDIDATE_QUESTIONS"
