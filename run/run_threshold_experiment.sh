#!/usr/bin/env bash


set -euo pipefail

# Usage: ./run/run_threshold_experiment.sh [--scene N] [--iterations N] [--threshold N]
# Example: ./run/run_threshold_experiment.sh --scene 3 --iterations 10 --threshold 0.8
THRESHOLD="0.8"
DOMAIN="wastesorting"
# DOMAIN="tomato"
SCENE="5"
ITERATIONS="10"
MAXSTEP="50"

usage() {
    echo "Usage: $0 [--scene N] [--iter N]"
    echo "  --scene N       scene number (default: ${SCENE})"
    echo "  --iteration  repeat count (default: ${ITERATIONS})"
    echo "  --threshold N   confidence threshold (default: ${THRESHOLD})"
}

while (($#)); do
    case "$1" in
        --scene)
            SCENE="${2:?Missing value for --scene}"
            shift 2
            ;;
        --iter|--iteration)
            ITERATIONS="${2:?Missing value for $1}"
            shift 2
            ;;
        --threshold)
            THRESHOLD="${2:?Missing value for --threshold}"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            usage
            exit 1
            ;;
    esac
done

if ! [[ "$SCENE" =~ ^[0-9]+$ && "$ITERATIONS" =~ ^[1-9][0-9]*$ ]]; then
    usage
    echo "  scene and iterations must be positive integers."
    exit 1
fi

THRESHOLD_LABEL="${THRESHOLD/./-}"
LOG_DIR="logs/${DOMAIN}/scene_0${SCENE}_step${MAXSTEP}/thres_${THRESHOLD_LABEL}"
# LOG_DIR="logs/${DOMAIN}/test/scene_0${SCENE}"


scene_id=$(printf "%02d" "$((10#$SCENE))")
initial_state="scripts/domain/${DOMAIN}/scene_${scene_id}.yaml"
domain_rule="scripts/domain/${DOMAIN}/domain_rule.yaml"
robot_skill="scripts/domain/${DOMAIN}/robot_skill.yaml"

if [[ ! -f "$initial_state" ]]; then
    echo "Initial state file not found: ${initial_state}"
    exit 1
fi
if [[ ! -f "$domain_rule" ]]; then
    echo "Domain rule file not found: ${domain_rule}"
    exit 1
fi
if [[ ! -f "$robot_skill" ]]; then
    echo "Robot skill file not found: ${robot_skill}"
    exit 1
fi

mkdir -p "$LOG_DIR"

for ((i = 1; i <= ITERATIONS; i++)); do
    seed=$(od -An -N4 -tu4 /dev/urandom | tr -d ' ')
    echo "[RUN ${i}/${ITERATIONS}] threshold=${THRESHOLD}, scene=${scene_id}, seed=${seed}, log_dir=${LOG_DIR}"

    python3 main.py \
        --domain "$DOMAIN" \
        --domain_rule "$domain_rule" \
        --initial_state "$initial_state" \
        --robot_skill "$robot_skill" \
        --threshold "$THRESHOLD" \
        --seed "$seed" \
        --log_dir "$LOG_DIR" \
        --max_step "$MAXSTEP"
done

echo "[DONE] Logs saved under ${LOG_DIR}"
