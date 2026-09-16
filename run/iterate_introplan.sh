#!/usr/bin/env bash
set -euo pipefail
# Compatibility entry point. Edit settings only in iterate_baseline.sh.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/iterate_baseline.sh" --baselines "introplan" "$@"
