#!/usr/bin/env bash
set -euo pipefail

# Compute KnowNo qhat values for the calibration datasets.
#
# Defaults:
#   - domains: tomato, wastesorting
#   - models: gpt-4o, gpt-3.5-turbo
#   - target success levels: 95%, 85%, 75%
#   - calibration size: 300 per domain
#   - score temperatures: 1.0, 3.0, 5.0
#
# Examples:
#   ./run/run_knowno_calibration.sh
#   MODELS="gpt-4o" ./run/run_knowno_calibration.sh
#   DOMAINS="tomato" TARGET_SUCCESSES="0.95,0.85,0.75" ./run/run_knowno_calibration.sh
#   TEMPERATURES="1 3 5" ./run/run_knowno_calibration.sh
#   NUM_CALIBRATION=200 ./run/run_knowno_calibration.sh
#   TOMATO_NUM_CALIBRATION=250 WASTE_NUM_CALIBRATION=300 ./run/run_knowno_calibration.sh
#   INCLUDE_PALM=true ./run/run_knowno_calibration.sh
#   SUMMARY_ONLY=true ./run/run_knowno_calibration.sh

# User-editable defaults. Environment variables with the same names override
# these values at runtime.
DEFAULT_NUM_CALIBRATION=100
DEFAULT_TOMATO_NUM_CALIBRATION="${DEFAULT_NUM_CALIBRATION}"
DEFAULT_WASTE_NUM_CALIBRATION="${DEFAULT_NUM_CALIBRATION}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CALIBRATION_SCRIPT="${PROJECT_ROOT}/scripts/baseline/compute_qhat.py"
SUMMARY_SCRIPT="${PROJECT_ROOT}/scripts/baseline/summarize_knowno_calibration.py"

DOMAINS="${DOMAINS:-tomato wastesorting}"
MODELS="${MODELS:-gpt-4o gpt-3.5-turbo}"
INCLUDE_PALM="${INCLUDE_PALM:-false}"
TARGET_SUCCESSES="${TARGET_SUCCESSES:-0.95,0.85,0.75}"
NUM_CALIBRATION="${NUM_CALIBRATION:-${DEFAULT_NUM_CALIBRATION}}"
TOMATO_NUM_CALIBRATION="${TOMATO_NUM_CALIBRATION:-${DEFAULT_TOMATO_NUM_CALIBRATION}}"
WASTE_NUM_CALIBRATION="${WASTE_NUM_CALIBRATION:-${DEFAULT_WASTE_NUM_CALIBRATION}}"
if [[ -z "${TEMPERATURES:-}" ]]; then
  if [[ -n "${TEMPERATURE:-}" ]]; then
    TEMPERATURES="${TEMPERATURE}"
  else
    TEMPERATURES="1 3 5"
  fi
fi
SETTINGS="${SETTINGS:-${PROJECT_ROOT}/scripts/baseline/llm_setting.json}"
API_KEY="${API_KEY:-}"
WRITE_SUMMARY="${WRITE_SUMMARY:-true}"
SUMMARY_OUTPUT="${SUMMARY_OUTPUT:-${PROJECT_ROOT}/experiments_logs/calibration_log/summary.md}"
SUMMARY_ONLY="${SUMMARY_ONLY:-false}"

if [[ "${INCLUDE_PALM}" == "true" ]]; then
  MODELS="${MODELS} PaLM-2L"
fi

cd "${PROJECT_ROOT}"

if [[ "${SUMMARY_ONLY}" != "true" ]]; then
  for domain in ${DOMAINS}; do
    if [[ "${domain}" == "tomato" ]]; then
      num_calibration="${TOMATO_NUM_CALIBRATION}"
    elif [[ "${domain}" == "wastesorting" || "${domain}" == "waste" ]]; then
      num_calibration="${WASTE_NUM_CALIBRATION}"
    else
      echo "Unsupported domain: ${domain}. Use tomato or wastesorting." >&2
      exit 1
    fi

    for model in ${MODELS}; do
      for temperature in ${TEMPERATURES}; do
        echo "============================================================"
        echo "KnowNo calibration: domain=${domain}, model=${model}, num_calibration=${num_calibration}"
        echo "Targets: ${TARGET_SUCCESSES}, temperature=${temperature}"
        echo "============================================================"

        cmd=(
          python3 "${CALIBRATION_SCRIPT}"
          --domain "${domain}"
          --num-calibration "${num_calibration}"
          --num-test 0
          --target-successes "${TARGET_SUCCESSES}"
          --temperature "${temperature}"
          --settings "${SETTINGS}"
          --model "${model}"
          --score-with-llm
        )

        if [[ -n "${API_KEY}" ]]; then
          cmd+=(--api-key "${API_KEY}")
        fi

        "${cmd[@]}"
      done
    done
  done

  echo "Calibration logs saved under ${PROJECT_ROOT}/experiments_logs/calibration_log"
fi

if [[ "${WRITE_SUMMARY}" == "true" ]]; then
  python3 "${SUMMARY_SCRIPT}" \
    --log-root "${PROJECT_ROOT}/experiments_logs/calibration_log" \
    --output "${SUMMARY_OUTPUT}"
fi
