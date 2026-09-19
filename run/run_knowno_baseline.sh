#!/usr/bin/env bash
set -euo pipefail

# 사용자 선택: tomato 또는 wastesorting
DOMAIN="${DOMAIN:-tomato}"

# 사용자 선택: 01, 02, 03, 04, 05
SCENE="${SCENE:-01}"

# 사용자 선택: v1(structured), v2(natural language)
PROMPT_VERSION="${PROMPT_VERSION:-v2}"

MAX_STEPS="${MAX_STEPS:-50}"
# Optional fixed seed. Leave empty to generate a new random seed on each run.
SEED="${SEED:-}"
# Lower temperature keeps option probabilities less flat, so the runner asks
# only when the model is genuinely uncertain.
SCORE_TEMPERATURE="${SCORE_TEMPERATURE:-5.0}"
VERBOSE="${VERBOSE:-false}"
DRY_RUN="${DRY_RUN:-false}"
LOG_FILE="${LOG_FILE:-}"
AUTO_ANSWER="${AUTO_ANSWER:-true}"

if [[ "${DOMAIN}" == "tomato" ]]; then
  QHAT="${QHAT:-0.8404}"
elif [[ "${DOMAIN}" == "wastesorting" || "${DOMAIN}" == "waste" ]]; then
  QHAT="${QHAT:-0.8704}"
else
  echo "Unsupported DOMAIN: ${DOMAIN}. Use tomato or wastesorting." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BASELINE_DIR="${PROJECT_ROOT}/scripts/baseline/knowno"
NORMALIZED_DOMAIN="${DOMAIN}"
[[ "${NORMALIZED_DOMAIN}" != "waste" ]] || NORMALIZED_DOMAIN="wastesorting"
ENV_SETTING="${ENV_SETTING:-${PROJECT_ROOT}/scripts/domain/${NORMALIZED_DOMAIN}/env_setting.yaml}"

if [[ -z "${SEED}" ]]; then
  SEED="$(( ( $(date +%s%N) + RANDOM ) % 4294967295 ))"
fi

CMD=(
  python3 "${BASELINE_DIR}/knowno_baseline_experiment.py"
  --domain "${DOMAIN}"
  --env-setting "${ENV_SETTING}"
  --scene "${SCENE}"
  --prompt-version "${PROMPT_VERSION}"
  --qhat "${QHAT}"
  --temperature "${SCORE_TEMPERATURE}"
  --max-steps "${MAX_STEPS}"
  --seed "${SEED}"
)

if [[ "${VERBOSE}" == "true" ]]; then
  CMD+=(--verbose)
fi

if [[ "${AUTO_ANSWER}" == "true" ]]; then
  CMD+=(--auto-answer)
fi

if [[ -n "${LOG_FILE}" ]]; then
  CMD+=(--log-file "${LOG_FILE}")
fi

if [[ "${DRY_RUN}" == "true" ]]; then
  CMD+=(--dry-run)
fi

cd "${PROJECT_ROOT}"
"${CMD[@]}"
