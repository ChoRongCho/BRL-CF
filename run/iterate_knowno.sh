#!/usr/bin/env bash

set -euo pipefail

# domains=(tomato wastesorting)
domains=(wastesorting tomato)
scenes=(1 2 3 4 5)
models=(gpt-4o gpt-3.5-turbo)
# models=(gpt-3.5-turbo)

qhat_targets=(95 85 75 raw98)

iterations="${ITERATIONS:-10}"
prompt_version="${PROMPT_VERSION:-v2}"
temperature="${TEMPERATURE:-5.0}"
max_steps="${MAX_STEPS:-50}"
seed="${SEED:-random}"
auto_answer="${AUTO_ANSWER:-true}"
settings_template="scripts/baseline/knowno/llm_setting.json"
log_root="experiments_logs/system_log"
seed_log_root="${log_root}/knowno_seed_logs"
seed_log="${seed_log_root}/iterate_knowno_$(date +%Y%m%d_%H%M%S).csv"

detect_success_prob="${DETECT_SUCCESS_PROB:-0.85}"
detect_label_error_prob="${DETECT_LABEL_ERROR_PROB:-0.05}"
scan_success_prob="${SCAN_SUCCESS_PROB:-0.9}"
scan_label_error_prob="${SCAN_LABEL_ERROR_PROB:-0.15}"
navigate_failure_prob="${NAVIGATE_FAILURE_PROB:-0.05}"
pick_failure_prob="${PICK_FAILURE_PROB:-0.05}"
place_failure_prob="${PLACE_FAILURE_PROB:-0.01}"
discard_failure_prob="${DISCARD_FAILURE_PROB:-0.01}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BASELINE_DIR="${PROJECT_ROOT}/scripts/baseline/knowno"

total=$((${#domains[@]} * ${#scenes[@]} * ${#models[@]} * ${#qhat_targets[@]} * iterations))
current=0
failed=0

mkdir -p "$seed_log_root"
echo "global_index,domain,scene,model,qhat_target,qhat,temperature,prompt_version,seed_mode,max_steps,auto_answer" > "$seed_log"

make_runtime_settings() {
    local model="$1"
    local output="$2"
    python3 - "${settings_template}" "${output}" "${model}" <<'PY'
import json
import sys
from pathlib import Path

src, dst, model = sys.argv[1:4]
settings = {}
path = Path(src)
if path.exists():
    settings.update(json.loads(path.read_text(encoding="utf-8")))
settings["model"] = model
Path(dst).write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
PY
}

qhat_for() {
    local domain="$1"
    local model="$2"
    local target="$3"

    case "${domain}:${model}:${target}" in
        tomato:gpt-3.5-turbo:95) echo "0.9243" ;;
        tomato:gpt-3.5-turbo:85) echo "0.9082" ;;
        tomato:gpt-3.5-turbo:75) echo "0.8938" ;;
        tomato:gpt-4o:95) echo "0.8404" ;;
        tomato:gpt-4o:85) echo "0.7779" ;;
        tomato:gpt-4o:75) echo "0.7322" ;;
        wastesorting:gpt-3.5-turbo:95) echo "0.9028" ;;
        wastesorting:gpt-3.5-turbo:85) echo "0.8851" ;;
        wastesorting:gpt-3.5-turbo:75) echo "0.8512" ;;
        wastesorting:gpt-4o:95) echo "0.8704" ;;
        wastesorting:gpt-4o:85) echo "0.7369" ;;
        wastesorting:gpt-4o:75) echo "0.7084" ;;
        *:*:raw98) echo "0.98" ;;
        *)
            echo "No qhat configured for domain=${domain}, model=${model}, target=${target}" >&2
            exit 1
            ;;
    esac
}

printf "\rProgress: %3d%%" 0

cd "${PROJECT_ROOT}"

for domain in "${domains[@]}"; do
    for scene in "${scenes[@]}"; do
        scene_id=$(printf "%02d" "$((10#$scene))")
        for model in "${models[@]}"; do
            runtime_settings="$(mktemp /tmp/knowno_iter_llm_setting_XXXXXX.json)"
            make_runtime_settings "$model" "$runtime_settings"
            trap 'rm -f "${runtime_settings}"' EXIT

            for qhat_target in "${qhat_targets[@]}"; do
                qhat="$(qhat_for "$domain" "$model" "$qhat_target")"
                for ((i = 1; i <= iterations; i++)); do
                    current=$((current + 1))
                    echo "${current},${domain},${scene_id},${model},${qhat_target},${qhat},${temperature},${prompt_version},${seed},${max_steps},${auto_answer}" >> "$seed_log"

                    cmd=(
                        python3 "${BASELINE_DIR}/knowno_baseline_experiment.py"
                        --domain "$domain"
                        --scene "$scene_id"
                        --settings "$runtime_settings"
                        --prompt-version "$prompt_version"
                        --qhat "$qhat"
                        --temperature "$temperature"
                        --max-steps "$max_steps"
                        --detect-success-prob "$detect_success_prob"
                        --detect-label-error-prob "$detect_label_error_prob"
                    )

                    if [[ "$auto_answer" == "true" ]]; then
                        cmd+=(--auto-answer)
                    fi

                    if [[ "$seed" != "random" ]]; then
                        cmd+=(--seed "$((seed + current - 1))")
                    fi

                    if [[ "$domain" == "tomato" ]]; then
                        cmd+=(
                            --scan-success-prob "$scan_success_prob"
                            --scan-label-error-prob "$scan_label_error_prob"
                            --navigate-failure-prob "$navigate_failure_prob"
                            --pick-failure-prob "$pick_failure_prob"
                            --place-failure-prob "$place_failure_prob"
                            --discard-failure-prob "$discard_failure_prob"
                        )
                    fi

                    if ! "${cmd[@]}" >/dev/null; then
                        failed=$((failed + 1))
                        echo "Run failed: domain=${domain}, scene=${scene_id}, model=${model}, qhat_target=${qhat_target}, qhat=${qhat}" >&2
                    fi
                    percent=$((current * 100 / total))
                    printf "\rProgress: %3d%%" "$percent"
                done
            done

            rm -f "$runtime_settings"
        done
    done
done

printf "\rProgress: 100%%\n"
echo "Seed mode log saved to ${seed_log}"
echo "Failed runs: ${failed}"
