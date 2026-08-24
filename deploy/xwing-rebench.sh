#!/usr/bin/env bash
# xwing re-bench — run AFTER finetune training finishes on xwing.
# Uses tuned configs (vulkan 2.28.2 pinned, ctx right-sized) from the
# projection-repair incident. See lms runs/model-shootout + FLEET_SWEEP docs.
set -u
XW=${XWING_SSH:-scott@192.168.1.86}   # LAN path; tailscale ssh blocked by check-mode
OUT="$HOME/git/lms/runs/fleet-maxprod-20260823"
echo "[1/4] clearing stale claims + reloading tuned models"
timeout 30 ssh -o BatchMode=yes $XWING_SSH '~/.lmstudio/bin/lms unload ternary-bonsai-27b@? >/dev/null 2>&1; true'
for m in ternary-bonsai-27b@? ornith-1.5-35b-a3b-apex-mtp; do
  timeout 300 ssh -o BatchMode=yes $XWING_SSH "~/.lmstudio/bin/lms load '$m' --gpu max -c 16384" 2>&1 | tail -1
done
echo "[2/4] concurrency sweep"
for c in 1 4 8; do
  python3 "$HOME/git/lms/runs/x1-370-gpu-maxprod-20260823/gpu_bench.py" \
    --base http://100.108.99.47:1234/v1 --model ornith-1.5-35b-a3b-apex-mtp \
    --conc $c --max-tokens 256 | tee -a "$OUT/xwing-rebench.jsonl"
  sleep 2
done
echo "[3/4] verify router sees fresh providers"
curl -s --max-time 8 http://127.0.0.1:8088/admin/runtime-projection >/dev/null 2>&1 || true
echo "[4/4] done — record results in $OUT"
