#!/bin/bash
# Waits for the clean bench (pid passed as $1) to finish, then refreshes loadout + report.
BPID="$1"
while kill -0 "$BPID" 2>/dev/null; do sleep 30; done
cd /home/scott/git/lms
echo "[hook] bench finished at $(date -u +%FT%TZ); refreshing plan + report" >> bench_clean_run.log
python3 fleet.py plan --demand balanced >> bench_clean_run.log 2>&1
python3 fleet.py report >> bench_clean_run.log 2>&1
echo "[hook] done. fresh loadout + report regenerated." >> bench_clean_run.log
