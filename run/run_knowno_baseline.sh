#!/usr/bin/env bash
set -euo pipefail

# 사용자 선택: tomato 또는 wastesorting
DOMAIN="${DOMAIN:-tomato}"

# 사용자 선택: 01, 02, 03, 04, 05
SCENE="${SCENE:-01}"

# 사용자 선택: v1(structured), v2(natural language)
PROMPT_VERSION="${PROMPT_VERSION:-v1}"

MAX_STEPS="${MAX_STEPS:-50}"
# Optional fixed seed. Leave empty to generate a new random seed on each run.
SEED="${SEED:-}"
# Lower temperature keeps option probabilities less flat, so the runner asks
# only when the model is genuinely uncertain.
SCORE_TEMPERATURE="${SCORE_TEMPERATURE:-3.0}"
VERBOSE="${VERBOSE:-false}"
DRY_RUN="${DRY_RUN:-false}"
LOG_FILE="${LOG_FILE:-}"

if [[ "${DOMAIN}" == "tomato" ]]; then
  QHAT="0.92"
  DETECT_SUCCESS_PROB="0.85"
  DETECT_LABEL_ERROR_PROB="0.05"
  SCAN_SUCCESS_PROB="0.9"
  SCAN_LABEL_ERROR_PROB="0.15"
  NAVIGATE_FAILURE_PROB="0.05"
  PICK_FAILURE_PROB="0.05"
  PLACE_FAILURE_PROB="0.01"
  DISCARD_FAILURE_PROB="0.01"
elif [[ "${DOMAIN}" == "wastesorting" || "${DOMAIN}" == "waste" ]]; then
  QHAT="0.92"
  DETECT_SUCCESS_PROB="0.9"
  DETECT_LABEL_ERROR_PROB="0.2"
else
  echo "Unsupported DOMAIN: ${DOMAIN}. Use tomato or wastesorting." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BASELINE_DIR="${PROJECT_ROOT}/scripts/baseline"

if [[ -z "${SEED}" ]]; then
  SEED="$(( ( $(date +%s%N) + RANDOM ) % 4294967295 ))"
fi

CMD=(
  python3 "${BASELINE_DIR}/knowno_baseline_experiment.py"
  --domain "${DOMAIN}"
  --scene "${SCENE}"
  --prompt-version "${PROMPT_VERSION}"
  --qhat "${QHAT}"
  --temperature "${SCORE_TEMPERATURE}"
  --max-steps "${MAX_STEPS}"
  --seed "${SEED}"
  --detect-success-prob "${DETECT_SUCCESS_PROB}"
  --detect-label-error-prob "${DETECT_LABEL_ERROR_PROB}"
)

if [[ "${DOMAIN}" == "tomato" ]]; then
  CMD+=(--scan-success-prob "${SCAN_SUCCESS_PROB}")
  CMD+=(--scan-label-error-prob "${SCAN_LABEL_ERROR_PROB}")
  CMD+=(--navigate-failure-prob "${NAVIGATE_FAILURE_PROB}")
  CMD+=(--pick-failure-prob "${PICK_FAILURE_PROB}")
  CMD+=(--place-failure-prob "${PLACE_FAILURE_PROB}")
  CMD+=(--discard-failure-prob "${DISCARD_FAILURE_PROB}")
fi

if [[ "${VERBOSE}" == "true" ]]; then
  CMD+=(--verbose)
fi

if [[ -n "${LOG_FILE}" ]]; then
  CMD+=(--log-file "${LOG_FILE}")
fi

if [[ "${DRY_RUN}" == "true" ]]; then
  CMD+=(--dry-run)
fi

cd "${PROJECT_ROOT}"
"${CMD[@]}"
