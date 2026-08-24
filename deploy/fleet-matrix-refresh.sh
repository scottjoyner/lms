#!/usr/bin/env bash
# Weekly fleet routing matrix refresh using latest committed benchmarks.
set -euo pipefail
export PATH="$HOME/git/lms/.venv/bin:$PATH"
set -a
[[ -f "$HOME/git/auto-assist/.env" ]] && source "$HOME/git/auto-assist/.env"
set +a
exec bash "$HOME/git/lms/scripts/refresh-fleet-routing-matrix.sh"
