#!/bin/bash
# Crash-doc benchmark pass: single-stream bench, then concurrency probe.
# LMS_REPO overrides the checkout location (no hardcoded personal path). W-71.
set -u
LMS_REPO="${LMS_REPO:-/home/scott/git/lms}"
cd "$LMS_REPO" || exit 1
echo "[$(date -u)] starting crash-doc pass (concurrency 9)" >> bench_fleet_run.log
python3 bench_fleet.py --concurrency 9 >> bench_fleet_run.log 2>&1
echo "[$(date -u)] crash-doc pass finished (rc=$?) -> launching concurrency probe" >> bench_concurrency_probe.log
python3 bench_concurrency_probe.py --max-concurrent 2 >> bench_concurrency_probe.log 2>&1
echo "[$(date -u)] concurrency probe done (rc=$?)" >> bench_concurrency_probe.log
