#!/usr/bin/env bash

set -euo pipefail

# Usage: ./run/run_scale_experiments.sh [--domain tomato|wastesorting] [--scene N] [--iter N] [--seed random|N]
# Scene 01-05 defaults to max_step=50 and max_belief_particles=800.
# Scene 06-10 defaults to max_step=90 and max_belief_particles=800.
# Scene 11-15 defaults to max_step=125 and max_belief_particles=8000.

DOMAIN="tomato"
SCENE="11"
ITERATIONS="1"
THRESHOLD="0.8"
SEED="random"
MAXSTEP=""
MAX_BELIEF_PARTICLES=""

usage() {
    echo "Usage: $0 [--domain tomato|wastesorting] [--scene N] [--iter N] [--threshold N] [--seed random|N] [--max-step N] [--max-belief-particles N]"
    echo "  --domain NAME          domain name (default: ${DOMAIN})"
    echo "  --scene N              scene number (default: ${SCENE})"
    echo "  --iter, --iteration N  repeat count (default: ${ITERATIONS})"
    echo "  --threshold N          confidence threshold (default: ${THRESHOLD})"
    echo "  --seed random|N        random per run or fixed integer seed (default: ${SEED})"
    echo "  --max-step N           override scale max step"
    echo "  --max-belief-particles N override scale belief particle cap"
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
        --max-step)
            MAXSTEP="${2:?Missing value for --max-step}"
            shift 2
            ;;
        --max-belief-particles)
            MAX_BELIEF_PARTICLES="${2:?Missing value for --max-belief-particles}"
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
if [[ -n "$MAXSTEP" && ! "$MAXSTEP" =~ ^[1-9][0-9]*$ ]]; then
    usage
    echo "  max-step must be a positive integer."
    exit 1
fi
if [[ -n "$MAX_BELIEF_PARTICLES" && ! "$MAX_BELIEF_PARTICLES" =~ ^[1-9][0-9]*$ ]]; then
    usage
    echo "  max-belief-particles must be a positive integer."
    exit 1
fi
if [[ "$SEED" != "random" && ! "$SEED" =~ ^[0-9]+$ ]]; then
    usage
    echo "  seed must be a non-negative integer or random."
    exit 1
fi

scene_num=$((10#$SCENE))
scene_id=$(printf "%02d" "$scene_num")
if [[ -z "$MAXSTEP" ]]; then
    if (( scene_num >= 6 && scene_num <= 10 )); then
        MAXSTEP="90"
    elif (( scene_num >= 11 && scene_num <= 15 )); then
        MAXSTEP="125"
    else
        MAXSTEP="50"
    fi
fi
if [[ -z "$MAX_BELIEF_PARTICLES" ]]; then
    if (( scene_num >= 6 && scene_num <= 10 )); then
        MAX_BELIEF_PARTICLES="800"
    elif (( scene_num >= 11 && scene_num <= 15 )); then
        MAX_BELIEF_PARTICLES="8000"
    else
        MAX_BELIEF_PARTICLES="8000"
    fi
fi

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

LOG_ROOT="experiments_logs/system_log"
LOG_DIR="${LOG_ROOT}/${DOMAIN}/scene_${scene_id}/scale_ours_step${MAXSTEP}"
mkdir -p "$LOG_DIR"

SEED_LOG="${LOG_DIR}/seeds.csv"
if [[ ! -f "$SEED_LOG" ]]; then
    echo "run_index,domain,scene,threshold,seed,max_step,max_belief_particles,log_dir" > "$SEED_LOG"
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
    echo "[RUN ${i}/${ITERATIONS}] domain=${DOMAIN}, scene=${scene_id}, threshold=${THRESHOLD}, seed=${seed}, max_step=${MAXSTEP}, max_belief_particles=${MAX_BELIEF_PARTICLES}, log_dir=${LOG_DIR}"
    echo "${i},${DOMAIN},${scene_id},${THRESHOLD},${seed},${MAXSTEP},${MAX_BELIEF_PARTICLES},${LOG_DIR}" >> "$SEED_LOG"

    python3 scale_main.py \
        --domain "$DOMAIN" \
        --domain_rule "$domain_rule" \
        --initial_state "$initial_state" \
        --robot_skill "$robot_skill" \
        --threshold "$THRESHOLD" \
        --seed "$seed" \
        --log_dir "$LOG_DIR" \
        --max_step "$MAXSTEP" \
        --max_belief_particles "$MAX_BELIEF_PARTICLES"
done

echo "[DONE] Scale logs saved under ${LOG_DIR}"
