#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

python_is_compatible() {
  "$1" -c 'import sys; raise SystemExit(sys.version_info < (3, 10))' >/dev/null 2>&1
}

if [[ -n "${PYTHON:-}" ]]; then
  PYTHON_BIN="$PYTHON"
  python_is_compatible "$PYTHON_BIN" || {
    echo "PYTHON must point to Python 3.10 or newer: $PYTHON_BIN" >&2
    exit 1
  }
elif python_is_compatible python3; then
  PYTHON_BIN="python3"
elif [[ -x /home/fr/miniconda3/envs/brl/bin/python ]] && python_is_compatible /home/fr/miniconda3/envs/brl/bin/python; then
  PYTHON_BIN="/home/fr/miniconda3/envs/brl/bin/python"
else
  echo "Python 3.10 or newer is required. Activate the brl environment or set PYTHON." >&2
  exit 1
fi

CONDITION="ours"
DOMAIN="tomato"
SCENE="1"
SEED="42"
MAX_STEP="50"
THRESHOLD="0.8"
N_SIMULATIONS="100"
MAX_DEPTH="20"
GAMMA="0.2"
UCB_C="1.0"
EPSILON="0.005"
MAX_PARTICLES="250"
MAX_BELIEF_PARTICLES="8000"
MAX_NODE_PARTICLES="8000"
QUERY_COST="1.0"
FAILURE_PENALTY="10.0"
ANSWER_ACCURACY="1.0"
SCORE_TEMPERATURE="5.0"
TOMATO_QHAT="0.8404"
WASTE_QHAT="0.8704"
LLM_SETTINGS="$PROJECT_ROOT/llm_setting.json"
LOG_DIR=""
DRY_RUN="false"

usage() {
  echo "Usage: $0 --condition ours|cp_when|value_when|value_what [options]"
  echo "  --domain tomato|wastesorting"
  echo "  --scene N --seed N --log-dir PATH --dry-run"
}

while (($#)); do
  case "$1" in
    --condition) CONDITION="${2:?Missing condition}"; shift 2 ;;
    --domain) DOMAIN="${2:?Missing domain}"; shift 2 ;;
    --scene) SCENE="${2:?Missing scene}"; shift 2 ;;
    --seed) SEED="${2:?Missing seed}"; shift 2 ;;
    --log-dir) LOG_DIR="${2:?Missing log directory}"; shift 2 ;;
    --max-step) MAX_STEP="${2:?Missing max step}"; shift 2 ;;
    --threshold) THRESHOLD="${2:?Missing threshold}"; shift 2 ;;
    --n-simulations) N_SIMULATIONS="${2:?Missing simulations}"; shift 2 ;;
    --gamma) GAMMA="${2:?Missing gamma}"; shift 2 ;;
    --dry-run) DRY_RUN="true"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

case "$CONDITION" in ours|cp_when|value_when|value_what) ;; *) echo "Unknown condition: $CONDITION" >&2; exit 1 ;; esac
case "$DOMAIN" in tomato|wastesorting) ;; *) echo "Unknown domain: $DOMAIN" >&2; exit 1 ;; esac
[[ "$SCENE" =~ ^[1-9][0-9]*$ ]] || { echo "scene must be positive" >&2; exit 1; }
[[ "$SEED" =~ ^(0|[1-9][0-9]*)$ ]] || { echo "seed must be non-negative" >&2; exit 1; }

scene_id=$(printf '%02d' "$((10#$SCENE))")
domain_root="$PROJECT_ROOT/scripts/domain/$DOMAIN"
initial_state="$domain_root/scene_${scene_id}.yaml"
domain_rule="$domain_root/domain_rule.yaml"
robot_skill="$domain_root/robot_skill.yaml"
env_setting="$domain_root/env_setting.yaml"
for path in "$initial_state" "$domain_rule" "$robot_skill" "$env_setting"; do
  [[ -f "$path" ]] || { echo "Missing input: $path" >&2; exit 1; }
done
if [[ -z "$LOG_DIR" ]]; then
  LOG_DIR="$PROJECT_ROOT/experiments_logs/when_what_mechanisms/draft/$DOMAIN/scene_${scene_id}/$CONDITION"
fi

command=("$PYTHON_BIN" -m scripts.ablation.when_what_mechanisms.runner
  --mechanism-condition "$CONDITION"
  --domain "$DOMAIN"
  --domain_rule "$domain_rule"
  --initial_state "$initial_state"
  --robot_skill "$robot_skill"
  --env-setting "$env_setting"
  --answer_type auto
  --seed "$SEED"
  --max_step "$MAX_STEP"
  --threshold "$THRESHOLD"
  --n_simulations "$N_SIMULATIONS"
  --max_depth "$MAX_DEPTH"
  --gamma "$GAMMA"
  --c "$UCB_C"
  --epsilon "$EPSILON"
  --max_particles "$MAX_PARTICLES"
  --max_belief_particles "$MAX_BELIEF_PARTICLES"
  --max_node_particles "$MAX_NODE_PARTICLES"
  --query-cost "$QUERY_COST"
  --failure-penalty "$FAILURE_PENALTY"
  --answer-accuracy "$ANSWER_ACCURACY"
  --score-temperature "$SCORE_TEMPERATURE"
  --tomato-qhat "$TOMATO_QHAT"
  --waste-qhat "$WASTE_QHAT"
  --llm-settings "$LLM_SETTINGS"
  --log_dir "$LOG_DIR")

cd "$PROJECT_ROOT"
if [[ "$DRY_RUN" == "true" ]]; then
  printf '%q ' "${command[@]}"; printf '\n'
  exit 0
fi
mkdir -p "$LOG_DIR"
exec "${command[@]}"
