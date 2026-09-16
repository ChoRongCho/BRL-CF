#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${PYTHON:-}" ]]; then
    GUI_PYTHON="$PYTHON"
elif [[ "${CONDA_DEFAULT_ENV:-}" == "brl" && -x "${CONDA_PREFIX:-}/bin/python" ]]; then
    GUI_PYTHON="$CONDA_PREFIX/bin/python"
elif [[ -x "$HOME/miniconda3/envs/brl/bin/python" ]]; then
    GUI_PYTHON="$HOME/miniconda3/envs/brl/bin/python"
else
    echo "brl conda 환경을 활성화한 뒤 실행하세요: conda activate brl" >&2
    echo "또는 PYTHON=/path/to/brl/bin/python 을 지정하세요." >&2
    exit 1
fi
# Existing baseline shell runners also invoke python3 through PATH.
export PATH="$(dirname "$GUI_PYTHON"):$PATH"
# Conda's Tk on this machine lacks Xft and cannot render installed CJK fonts.
# The GUI uses only the standard library; workers still use brl via PATH.
if [[ -x /usr/bin/python3 ]] && /usr/bin/python3 -c 'import tkinter' 2>/dev/null; then
    exec /usr/bin/python3 "$SCRIPT_DIR/run_baseline_gui.py" "$@"
fi
exec "$GUI_PYTHON" "$SCRIPT_DIR/run_baseline_gui.py" "$@"
