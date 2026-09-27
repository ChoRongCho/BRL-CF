#!/usr/bin/env bash
set -euo pipefail

# ======================== Baseline experiment settings ========================
# Edit only this block. Settings for all three baselines live here.
BASELINES=(knowno introplan query_action_pomcp)
DOMAINS=(tomato wastesorting)
SCENES=(1 2 3 4 5)
# Per baseline, domain, and scene: 3 x 2 x 5 x 40 = 1200 total episodes.
ITERATIONS_PER_SCENE="40"
MAX_STEPS="50"
PAIRED_SEED_LOG="experiments_logs/system_log/when_what_seed_logs/iterate_when_what_20260912_150132.csv"

# KnowNo and IntroPlan (same action oracle).
PROMPT_VERSION="v2"
SCORE_TEMPERATURE="5.0"
KNOWNO_TOMATO_QHAT="0.8404"
KNOWNO_WASTE_QHAT="0.8704"
INTROPLAN_TOMATO_QHAT="0.9809474992495626"
INTROPLAN_WASTE_QHAT="0.9615342162270937"
TOP_K="3"
KNOWLEDGE_FILE="scripts/baseline/introplan/knowledge.json"

# Query-as-Action POMCP (Boolean fact oracle).
N_SIMULATIONS="100"
MAX_DEPTH="20"
GAMMA="0.2" # Backward-compatible shared value used by --gamma.
TOMATO_GAMMA="0.5"
WASTE_GAMMA="0.9"
UCB_C="1.0"
EPSILON="0.005"
MAX_PARTICLES="250"
MAX_BELIEF_PARTICLES="8000"
MAX_NODE_PARTICLES="8000"
QUERY_COST="0.0"
FAILURE_PENALTY="10.0"
ANSWER_ACCURACY="1.0"

LOG_ROOT="experiments_logs/system_log"
ARCHIVE_EXISTING="true" # Archive only selected baseline/domain/scene folders.
RESUME="false"          # Resume disables archival.
DRY_RUN="false"         # Validate every seed; no API calls or file changes.
# =============================================================================

usage() {
    echo "Usage: $0 [--dry-run] [--resume] [--no-archive] [--iter N] [--baseline NAME]"
    echo 'Select several methods: --baselines "knowno introplan query_action_pomcp"'
    echo 'Select domains: --domains "tomato wastesorting"'
    echo "CP overrides: --knowno-waste-qhat X --introplan-waste-qhat X"
    echo "Query-as-Action overrides: --gamma X or --tomato-gamma X --waste-gamma X"
    echo "                           --query-cost X --failure-penalty X"
    echo "Output override: --log-root PATH"
    echo "Edit the settings block in this file for persistent changes."
}
while (($#)); do
    case "$1" in
        --dry-run) DRY_RUN="true"; shift ;;
        --resume) RESUME="true"; ARCHIVE_EXISTING="false"; shift ;;
        --no-archive) ARCHIVE_EXISTING="false"; shift ;;
        --iter|--iteration) ITERATIONS_PER_SCENE="${2:?Missing iteration count}"; shift 2 ;;
        --baseline|--baselines) read -r -a BASELINES <<< "${2:?Missing baseline name}"; shift 2 ;;
        --domain|--domains) read -r -a DOMAINS <<< "${2:?Missing domain name}"; shift 2 ;;
        --knowno-tomato-qhat) KNOWNO_TOMATO_QHAT="${2:?Missing qhat}"; shift 2 ;;
        --knowno-waste-qhat) KNOWNO_WASTE_QHAT="${2:?Missing qhat}"; shift 2 ;;
        --introplan-tomato-qhat) INTROPLAN_TOMATO_QHAT="${2:?Missing qhat}"; shift 2 ;;
        --introplan-waste-qhat) INTROPLAN_WASTE_QHAT="${2:?Missing qhat}"; shift 2 ;;
        --gamma)
            GAMMA="${2:?Missing gamma}"
            TOMATO_GAMMA="$GAMMA"
            WASTE_GAMMA="$GAMMA"
            shift 2 ;;
        --tomato-gamma) TOMATO_GAMMA="${2:?Missing tomato gamma}"; shift 2 ;;
        --waste-gamma) WASTE_GAMMA="${2:?Missing waste gamma}"; shift 2 ;;
        --query-cost) QUERY_COST="${2:?Missing query cost}"; shift 2 ;;
        --failure-penalty) FAILURE_PENALTY="${2:?Missing failure penalty}"; shift 2 ;;
        --log-root) LOG_ROOT="${2:?Missing log root}"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
    esac
done
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export BATCH_BASELINES="${BASELINES[*]}" BATCH_DOMAINS="${DOMAINS[*]}" BATCH_SCENES="${SCENES[*]}"
export BATCH_ITERATIONS="$ITERATIONS_PER_SCENE"
for name in MAX_STEPS PAIRED_SEED_LOG PROMPT_VERSION SCORE_TEMPERATURE \
    KNOWNO_TOMATO_QHAT KNOWNO_WASTE_QHAT INTROPLAN_TOMATO_QHAT INTROPLAN_WASTE_QHAT \
    TOP_K KNOWLEDGE_FILE N_SIMULATIONS MAX_DEPTH GAMMA TOMATO_GAMMA WASTE_GAMMA \
    UCB_C EPSILON MAX_PARTICLES \
    MAX_BELIEF_PARTICLES MAX_NODE_PARTICLES QUERY_COST FAILURE_PENALTY ANSWER_ACCURACY \
    LOG_ROOT ARCHIVE_EXISTING RESUME DRY_RUN; do
    export "BATCH_${name}=${!name}"
done
exec python3 "$SCRIPT_DIR/../scripts/baseline/iterate.py"
