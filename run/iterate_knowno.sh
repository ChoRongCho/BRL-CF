#!/usr/bin/env bash

set -euo pipefail

# ======================== KnowNo experiment settings ========================
# Edit only this block for the next batch.

# Each selected domain is run for every selected scene.
DOMAINS=(tomato wastesorting)
SCENES=(1 2 3 4 5)

# Repetitions PER DOMAIN AND PER SCENE.
# Current setting: 2 domains x 5 scenes x 40 = 400 total episodes
#                 5 scenes x 40 = 200 episodes per domain
ITERATIONS_PER_SCENE="40"
MAX_STEPS="50"

# Use the exact same seeds as the completed Ours When+What experiment.
# The file currently contains 40 seeds for every domain/scene pair, so
# ITERATIONS_PER_SCENE may be set from 1 through 40 with this seed file.
PAIRED_SEED_LOG="experiments_logs/system_log/when_what_seed_logs/iterate_when_what_20260912_150132.csv"

# KnowNo parameters.  These reproduce the historical v2 configuration while
# using the current tomato ripeness/freshness domain representation.
PROMPT_VERSION="v2"
SCORE_TEMPERATURE="5.0"
TOMATO_QHAT="0.8404"
WASTE_QHAT="0.8704"

# Output/control settings.
LOG_ROOT="experiments_logs/system_log"
ARCHIVE_EXISTING="true"  # Move an existing KnowNo result aside before running.
RESUME="false"           # true: skip episodes already marked complete.
DRY_RUN="false"          # true: validate and print commands without API calls.

# ===========================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    echo "Usage: $0 [--dry-run] [--resume] [--no-archive] [--iter N]"
    echo "The editable settings block at the top is used by default."
}

# Command-line options override the editable values above for one invocation.
while (($#)); do
    case "$1" in
        --dry-run) DRY_RUN="true"; shift ;;
        --resume) RESUME="true"; ARCHIVE_EXISTING="false"; shift ;;
        --no-archive) ARCHIVE_EXISTING="false"; shift ;;
        --iter|--iteration)
            ITERATIONS_PER_SCENE="${2:?Missing value for $1}"
            shift 2
            ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
    esac
done

if ((${#DOMAINS[@]} == 0 || ${#SCENES[@]} == 0)); then
    echo "DOMAINS and SCENES must not be empty." >&2
    exit 1
fi

if ! [[ "$ITERATIONS_PER_SCENE" =~ ^[1-9][0-9]*$ ]]; then
    echo "ITERATIONS_PER_SCENE must be a positive integer." >&2
    exit 1
fi

total=$((${#DOMAINS[@]} * ${#SCENES[@]} * ITERATIONS_PER_SCENE))
echo "KnowNo batch configuration"
echo "  domains: ${DOMAINS[*]}"
echo "  scenes: ${SCENES[*]}"
echo "  repetitions per domain/scene: ${ITERATIONS_PER_SCENE}"
echo "  total episodes: ${total}"
echo "  prompt: ${PROMPT_VERSION}"
echo "  dry run: ${DRY_RUN}"

BATCH_DOMAINS="${DOMAINS[*]}" \
BATCH_SCENES="${SCENES[*]}" \
BATCH_ITERATIONS="$ITERATIONS_PER_SCENE" \
BATCH_MAX_STEPS="$MAX_STEPS" \
BATCH_PAIRED_SEED_LOG="$PAIRED_SEED_LOG" \
BATCH_LOG_ROOT="$LOG_ROOT" \
BATCH_PROMPT_VERSION="$PROMPT_VERSION" \
BATCH_SCORE_TEMPERATURE="$SCORE_TEMPERATURE" \
BATCH_TOMATO_QHAT="$TOMATO_QHAT" \
BATCH_WASTE_QHAT="$WASTE_QHAT" \
BATCH_ARCHIVE_EXISTING="$ARCHIVE_EXISTING" \
BATCH_RESUME="$RESUME" \
BATCH_DRY_RUN="$DRY_RUN" \
    exec "$SCRIPT_DIR/iterate_query_baselines.sh" --baseline knowno
