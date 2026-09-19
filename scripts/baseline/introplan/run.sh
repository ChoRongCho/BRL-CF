#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARGS=(--domain "${DOMAIN:-tomato}" --scene "${SCENE:-01}"
      --max-steps "${MAX_STEPS:-50}" --temperature "${SCORE_TEMPERATURE:-5.0}"
      --prompt-version "${PROMPT_VERSION:-v2}" --top-k "${TOP_K:-3}" --auto-answer)
[[ -z "${ENV_SETTING:-}" ]] || ARGS+=(--env-setting "$ENV_SETTING")
[[ -z "${KNOWLEDGE_FILE:-}" ]] || ARGS+=(--knowledge "$KNOWLEDGE_FILE")
[[ -z "${LOG_FILE:-}" ]] || ARGS+=(--log-file "$LOG_FILE")
[[ -z "${SEED:-}" ]] || ARGS+=(--seed "$SEED")
[[ -z "${QHAT:-}" ]] || ARGS+=(--qhat "$QHAT")
[[ "${DRY_RUN:-false}" != true ]] || ARGS+=(--dry-run)
[[ "${VERBOSE:-true}" != true ]] || ARGS+=(--verbose)
exec python3 "$SCRIPT_DIR/run_experiment.py" "${ARGS[@]}" "$@"
