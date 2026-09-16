#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARGS=(--domain "${DOMAIN:-tomato}" --scene "${SCENE:-01}"
      --max-steps "${MAX_STEPS:-50}" --temperature "${SCORE_TEMPERATURE:-5.0}"
      --prompt-version "${PROMPT_VERSION:-v2}" --top-k "${TOP_K:-3}" --auto-answer)
[[ -z "${KNOWLEDGE_FILE:-}" ]] || ARGS+=(--knowledge "$KNOWLEDGE_FILE")
[[ -z "${LOG_FILE:-}" ]] || ARGS+=(--log-file "$LOG_FILE")
[[ -z "${SEED:-}" ]] || ARGS+=(--seed "$SEED")
[[ -z "${QHAT:-}" ]] || ARGS+=(--qhat "$QHAT")
[[ "${DRY_RUN:-false}" != true ]] || ARGS+=(--dry-run)
[[ "${VERBOSE:-true}" != true ]] || ARGS+=(--verbose)
# Match the KnowNo comparison runner's environment noise.
if [[ "${DOMAIN:-tomato}" == tomato ]]; then
    ARGS+=(--detect-success-prob 0.85 --detect-label-error-prob 0.05
           --scan-success-prob 0.9 --scan-label-error-prob 0.15
           --navigate-failure-prob 0.05 --pick-failure-prob 0.05
           --place-failure-prob 0.01 --discard-failure-prob 0.01)
else
    ARGS+=(--detect-success-prob 0.9 --detect-label-error-prob 0.2)
fi
exec python3 "$SCRIPT_DIR/run_experiment.py" "${ARGS[@]}" "$@"
