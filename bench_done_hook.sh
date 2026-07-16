#!/bin/bash
# Waits for the clean bench (pid passed as $1) to finish, then refreshes loadout + report.
#
# Paths are parameterized: set LMS_REPO to the lms checkout (defaults to the
# original location). No hardcoded personal paths/keys. See W-71
# (docs/LLD_UNIFIED_FLEET.md).
set -u
BPID="${1:?usage: bench_done_hook.sh <bench_pid>}"
LMS_REPO="${LMS_REPO:-/home/scott/git/lms}"
cd "$LMS_REPO" || exit 1
echo "[hook] bench finished at $(date -u +%FT%TZ); refreshing plan + report" >> bench_clean_run.log
python3 fleet.py plan --demand balanced >> bench_clean_run.log 2>&1
python3 fleet.py report >> bench_clean_run.log 2>&1
echo "[hook] done. fresh loadout + report regenerated." >> bench_clean_run.log
