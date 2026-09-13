#!/usr/bin/env bash

set -euo pipefail

# One-episode entry point shared by both query baselines.
# BASELINE: knowno | query_action_pomcp
BASELINE="${BASELINE:-knowno}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

case "$BASELINE" in
    knowno)
        exec "$SCRIPT_DIR/run_knowno_baseline.sh"
        ;;
    query_action_pomcp|query-as-action|targeted_query_pomdp)
        TARGET_RUN="$SCRIPT_DIR/../scripts/baseline/targeted_query_pomdp/run.sh"
        if [[ "${DRY_RUN:-false}" == "true" ]]; then
            printf 'BASELINE=query_action_pomcp DOMAIN=%q SCENE=%q SEED=%q QUERY_COST=%q FAILURE_PENALTY=%q N_SIMULATIONS=%q MAX_DEPTH=%q GAMMA=%q UCB_C=%q EPSILON=%q MAX_PARTICLES=%q MAX_BELIEF_PARTICLES=%q MAX_NODE_PARTICLES=%q %q\n' \
                "${DOMAIN:-tomato}" "${SCENE:-01}" "${SEED:-random}" \
                "${QUERY_COST:-1.0}" "${FAILURE_PENALTY:-10.0}" "${N_SIMULATIONS:-100}" "${MAX_DEPTH:-20}" \
                "${GAMMA:-0.95}" "${UCB_C:-1.0}" "${EPSILON:-0.005}" \
                "${MAX_PARTICLES:-250}" "${MAX_BELIEF_PARTICLES:-8000}" "${MAX_NODE_PARTICLES:-8000}" \
                "$TARGET_RUN"
            exit 0
        fi
        exec "$TARGET_RUN"
        ;;
    *)
        echo "Unsupported BASELINE: $BASELINE" >&2
        echo "Use knowno or query_action_pomcp." >&2
        exit 1
        ;;
esac
