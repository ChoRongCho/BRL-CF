#!/usr/bin/env bash

# Shared settings and runners for the experiment plan in 10_TODO_20260727.md.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOMAINS="${DOMAINS:-tomato wastesorting}"
SCENES="${SCENES:-1 2 3 4 5}"
ITERATIONS="${ITERATIONS:-40}"
MAX_STEP="${MAX_STEP:-50}"
TAU_GLOBAL="${TAU_GLOBAL:-0.8}"
TOMATO_RANDOM_QUERY_PROB="${TOMATO_RANDOM_QUERY_PROB:-0.48}"
WASTESORTING_RANDOM_QUERY_PROB="${WASTESORTING_RANDOM_QUERY_PROB:-0.38}"
KNOWNO_SETTINGS="${KNOWNO_SETTINGS:-$ROOT/llm_setting_dummy.json}"
KNOWNO_SCORE_TEMPERATURE="${KNOWNO_SCORE_TEMPERATURE:-5.0}"
KNOWNO_QHAT_TOMATO="${KNOWNO_QHAT_TOMATO:-0.7779}"
KNOWNO_QHAT_WASTESORTING="${KNOWNO_QHAT_WASTESORTING:-0.7369}"
BASE_SEED="${BASE_SEED:-20260727}"
EXPERIMENT_LOG_ROOT="${EXPERIMENT_LOG_ROOT:-$ROOT/experiments_logs/system_log}"

# The same domain/scene/run tuple receives the same seed across methods.
experiment_seed() {
    local domain="$1" scene="$2" run_index="$3"
    local domain_offset=0
    [[ "$domain" == "wastesorting" ]] && domain_offset=100000
    printf "%d" "$((BASE_SEED + domain_offset + 10#$scene * 1000 + 10#$run_index))"
}

scene_id() {
    printf "%02d" "$((10#$1))"
}

experiment_log_dir() {
    local experiment="$1" domain="$2" scene="$3" method="$4" feedback_source="$5"
    printf "%s/%s/%s/scene_%s_step%s/%s/%s" \
        "$EXPERIMENT_LOG_ROOT" \
        "$experiment" \
        "$domain" \
        "$(scene_id "$scene")" \
        "$MAX_STEP" \
        "$method" \
        "$feedback_source"
}

word_count() {
    local values="$1"
    local -a items=()
    read -r -a items <<< "$values"
    printf "%d" "${#items[@]}"
}

random_query_prob() {
    local domain="$1"
    case "$domain" in
        tomato) printf "%s" "$TOMATO_RANDOM_QUERY_PROB" ;;
        wastesorting) printf "%s" "$WASTESORTING_RANDOM_QUERY_PROB" ;;
        *) echo "No Random-When query probability configured for domain: $domain" >&2; return 1 ;;
    esac
}

knowno_qhat() {
    local domain="$1"
    case "$domain" in
        tomato) printf "%s" "$KNOWNO_QHAT_TOMATO" ;;
        wastesorting) printf "%s" "$KNOWNO_QHAT_WASTESORTING" ;;
        *) echo "No KnowNo qhat configured for domain: $domain" >&2; return 1 ;;
    esac
}

progress_update() {
    local completed="$1" total="$2" label="${3:-}"
    local width=30 percent filled empty bar

    if ((total <= 0)); then
        percent=100
        filled=$width
    else
        percent=$((completed * 100 / total))
        filled=$((completed * width / total))
    fi
    empty=$((width - filled))
    printf -v bar '%*s' "$filled" ''
    bar="${bar// /#}"
    printf -v empty_bar '%*s' "$empty" ''
    empty_bar="${empty_bar// /-}"
    if [[ -t 1 ]]; then
        # Clear the current line first so shorter labels do not leave residue.
        # Do not pad the label: a wide line wraps and leaves one line per update.
        printf '\r\033[K[%s%s] %3d%% (%d/%d) %s' \
            "$bar" "$empty_bar" "$percent" "$completed" "$total" "$label"
        if ((completed >= total)); then
            printf '\n'
        fi
    else
        # Redirected/CI logs only need percentage changes, not one line per run.
        if [[ "${LAST_PROGRESS_PERCENT:-}" == "$percent" && completed -lt total ]]; then
            return
        fi
        LAST_PROGRESS_PERCENT="$percent"
        printf '[%s%s] %3d%% (%d/%d) %s\n' \
            "$bar" "$empty_bar" "$percent" "$completed" "$total" "$label"
    fi
}

run_quietly() {
    local console_log="$1"
    shift
    mkdir -p "$(dirname "$console_log")"
    if ! "$@" >"$console_log" 2>&1; then
        echo "Experiment failed. Last 40 lines from $console_log:" >&2
        tail -n 40 "$console_log" >&2
        return 1
    fi
}

run_ours() {
    local domain="$1" scene="$2" threshold="$3" answer_type="$4" run_seed="$5" log_dir="$6"
    local id
    id="$(scene_id "$scene")"
    mkdir -p "$log_dir"
    local -a command=(
        env FEEDBACK_METHOD=ours python3 "$ROOT/main.py"
        --domain "$domain"
        --domain_rule "$ROOT/scripts/domain/$domain/domain_rule.yaml"
        --initial_state "$ROOT/scripts/domain/$domain/scene_$id.yaml"
        --robot_skill "$ROOT/scripts/domain/$domain/robot_skill.yaml"
        --threshold "$threshold"
        --answer_type "$answer_type"
        --seed "$run_seed"
        --log_dir "$log_dir"
        --max_step "$MAX_STEP"
    )
    if [[ "$answer_type" == "human" ]]; then
        "${command[@]}"
    else
        run_quietly "$log_dir/console_seed_${run_seed}.log" "${command[@]}"
    fi
}

run_random_when() {
    local domain="$1" scene="$2" run_seed="$3" log_dir="$4"
    local id query_prob
    id="$(scene_id "$scene")"
    query_prob="$(random_query_prob "$domain")"
    mkdir -p "$log_dir"
    run_quietly "$log_dir/console_seed_${run_seed}.log" \
        env FEEDBACK_METHOD=ours-random-when python3 "$ROOT/main.py" \
            --domain "$domain" \
            --domain_rule "$ROOT/scripts/domain/$domain/domain_rule.yaml" \
            --initial_state "$ROOT/scripts/domain/$domain/scene_$id.yaml" \
            --robot_skill "$ROOT/scripts/domain/$domain/robot_skill.yaml" \
            --threshold "$TAU_GLOBAL" \
            --random_query_prob "$query_prob" \
            --answer_type oracle \
            --seed "$run_seed" \
            --log_dir "$log_dir" \
            --max_step "$MAX_STEP"
}

run_random_what() {
    local domain="$1" scene="$2" run_seed="$3" log_dir="$4"
    local id
    id="$(scene_id "$scene")"
    mkdir -p "$log_dir"
    run_quietly "$log_dir/console_seed_${run_seed}.log" \
        env FEEDBACK_METHOD=ours-random-what python3 "$ROOT/main.py" \
            --domain "$domain" \
            --domain_rule "$ROOT/scripts/domain/$domain/domain_rule.yaml" \
            --initial_state "$ROOT/scripts/domain/$domain/scene_$id.yaml" \
            --robot_skill "$ROOT/scripts/domain/$domain/robot_skill.yaml" \
            --threshold "$TAU_GLOBAL" \
            --answer_type oracle \
            --seed "$run_seed" \
            --log_dir "$log_dir" \
            --max_step "$MAX_STEP"
}

run_knowno() {
    local domain="$1" scene="$2" answer_type="$3" run_seed="$4" log_file="$5"
    local qhat
    qhat="$(knowno_qhat "$domain")"
    local -a answer_args=(--answer-type "$answer_type")
    if [[ "$answer_type" != "human" ]]; then
        answer_args+=(--auto-answer)
    fi
    mkdir -p "$(dirname "$log_file")"
    local -a command=(
        python3 "$ROOT/scripts/baseline/knowno/runners/knowno_baseline_experiment.py"
        --domain "$domain"
        --scene "$(scene_id "$scene")"
        --settings "$KNOWNO_SETTINGS"
        --qhat "$qhat"
        --score-temperature "$KNOWNO_SCORE_TEMPERATURE"
        --seed "$run_seed"
        --max-steps "$MAX_STEP"
        --log-file "$log_file"
        "${answer_args[@]}"
    )
    if [[ "$answer_type" == "human" ]]; then
        "${command[@]}"
    else
        run_quietly "${log_file%.json}.console.log" "${command[@]}"
    fi
}
