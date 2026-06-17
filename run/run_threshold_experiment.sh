#!/usr/bin/env bash


set -euo pipefail

# Usage: ./run/run_threshold_experiment.sh [--domain tomato|wastesorting] [--scene N] [--iterations N] [--threshold N] [--seed random|N]
# Example: ./run/run_threshold_experiment.sh --domain tomato --scene 3 --iterations 10 --threshold 0.8 --seed random
THRESHOLD="0.8"
DOMAIN="wastesorting"
# DOMAIN="tomato"
SCENE="5"
ITERATIONS="10"
MAXSTEP="50"
SEED="random"

usage() {
    echo "Usage: $0 [--domain tomato|wastesorting] [--scene N] [--iter N] [--seed random|N]"
    echo "  --domain NAME   domain name (default: ${DOMAIN})"
    echo "  --scene N       scene number (default: ${SCENE})"
    echo "  --iteration  repeat count (default: ${ITERATIONS})"
    echo "  --threshold N   confidence threshold (default: ${THRESHOLD})"
    echo "  --seed random|N random per run or fixed integer seed (default: ${SEED})"
}

while (($#)); do
    case "$1" in
        --domain)
            DOMAIN="${2:?Missing value for --domain}"
            shift 2
            ;;
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
        --seed)
            SEED="${2:?Missing value for --seed}"
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
if [[ "$SEED" != "random" && ! "$SEED" =~ ^[0-9]+$ ]]; then
    usage
    echo "  seed must be a non-negative integer or random."
    exit 1
fi

THRESHOLD_LABEL="${THRESHOLD/./-}"
scene_id=$(printf "%02d" "$((10#$SCENE))")
LOG_ROOT="experiments_logs/system_log"
LOG_DIR="${LOG_ROOT}/${DOMAIN}/scene_${scene_id}_step${MAXSTEP}/thres_${THRESHOLD_LABEL}"

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
SEED_LOG="${LOG_DIR}/seeds.csv"
if [[ ! -f "$SEED_LOG" ]]; then
    echo "run_index,domain,scene,threshold,seed,log_dir" > "$SEED_LOG"
fi

generate_seed() {
    od -An -N4 -tu4 /dev/urandom | tr -d ' '
}

for ((i = 1; i <= ITERATIONS; i++)); do
    if [[ "$SEED" == "random" ]]; then
        seed=$(generate_seed)
    else
        seed="$SEED"
    fi
    echo "[RUN ${i}/${ITERATIONS}] threshold=${THRESHOLD}, scene=${scene_id}, seed=${seed}, log_dir=${LOG_DIR}"
    echo "${i},${DOMAIN},${scene_id},${THRESHOLD},${seed},${LOG_DIR}" >> "$SEED_LOG"

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
