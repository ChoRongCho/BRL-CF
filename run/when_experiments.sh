#!/usr/bin/env bash

set -euo pipefail

# Usage: ./run/when_experiments.sh [--scene N] [--iter N] [--strategy no|all|ours|random] [--random-query-prob P] [--seed N|random]
# Example: ./run/when_experiments.sh --scene 3 --iter 10 --strategy random --random-query-prob 0.3 --seed random
DOMAIN="tomato"
# DOMAIN="wastesorting"
SCENE="5"
ITERATIONS="1"
MAXSTEP="50"
STRATEGY="random"
RANDOM_QUERY_PROB="0.3"
THRESHOLD="0.8"
SEED="random"

usage() {
    echo "Usage: $0 [--scene N] [--iter N] [--strategy no|all|ours|random] [--random-query-prob P] [--seed N|random]"
    echo "  --scene N              scene number (default: ${SCENE})"
    echo "  --iter, --iteration N  repeat count (default: ${ITERATIONS})"
    echo "  --strategy NAME        query policy: no, all, ours, random (default: ${STRATEGY})"
    echo "  --random-query-prob P  trigger probability for random strategy (default: ${RANDOM_QUERY_PROB})"
    echo "  --seed N|random        fixed seed or random seed per run (default: ${SEED})"
    echo "  --domain NAME          domain name (default: ${DOMAIN})"
    echo "  --max-step N           maximum episode steps (default: ${MAXSTEP})"
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
        --strategy)
            STRATEGY="${2:?Missing value for --strategy}"
            shift 2
            ;;
        --random-query-prob)
            RANDOM_QUERY_PROB="${2:?Missing value for --random-query-prob}"
            shift 2
            ;;
        --seed)
            SEED="${2:?Missing value for --seed}"
            shift 2
            ;;
        --domain)
            DOMAIN="${2:?Missing value for --domain}"
            shift 2
            ;;
        --max-step)
            MAXSTEP="${2:?Missing value for --max-step}"
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

if ! [[ "$SCENE" =~ ^[0-9]+$ && "$ITERATIONS" =~ ^[1-9][0-9]*$ && "$MAXSTEP" =~ ^[1-9][0-9]*$ ]]; then
    usage
    echo "  scene, iterations, and max-step must be positive integers."
    exit 1
fi
if [[ "$SEED" != "random" && ! "$SEED" =~ ^[0-9]+$ ]]; then
    usage
    echo "  seed must be a non-negative integer or random."
    exit 1
fi

case "$STRATEGY" in
    no)
        F_STRATEGY="1"
        ;;
    all)
        F_STRATEGY="2"
        ;;
    ours)
        F_STRATEGY="3"
        ;;
    random)
        F_STRATEGY="4"
        ;;
    *)
        usage
        echo "  unknown strategy: ${STRATEGY}"
        exit 1
        ;;
esac

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

PROB_LABEL="${RANDOM_QUERY_PROB/./-}"
LOG_ROOT="experiments_logs/system_log"
LOG_DIR="${LOG_ROOT}/${DOMAIN}/scene_${scene_id}_step${MAXSTEP}/when_${STRATEGY}_rand_${PROB_LABEL}"

mkdir -p "$LOG_DIR"

generate_seed() {
    od -An -N4 -tu4 /dev/urandom | tr -d ' '
}

for ((i = 1; i <= ITERATIONS; i++)); do
    if [[ "$SEED" == "random" ]]; then
        seed=$(generate_seed)
    else
        seed="$SEED"
    fi
    echo "[RUN ${i}/${ITERATIONS}] strategy=${STRATEGY}, threshold=${THRESHOLD}, random_query_prob=${RANDOM_QUERY_PROB}, scene=${scene_id}, seed=${seed}, log_dir=${LOG_DIR}"

    python3 when_main.py \
        --domain "$DOMAIN" \
        --domain_rule "$domain_rule" \
        --initial_state "$initial_state" \
        --robot_skill "$robot_skill" \
        --f_strategy "$F_STRATEGY" \
        --random_query_prob "$RANDOM_QUERY_PROB" \
        --seed "$seed" \
        --log_dir "$LOG_DIR" \
        --max_step "$MAXSTEP"
done

echo "[DONE] Logs saved under ${LOG_DIR}"
