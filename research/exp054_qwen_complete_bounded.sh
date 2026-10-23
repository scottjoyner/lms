#!/usr/bin/env bash
# EXP-054 exact BF16 Qwen model scoring, source-custody bound machine-only.
# Operator-initiated and foreground, never a background/production deployment.
set -euo pipefail
repo=/home/scott/git/wt-lms-exp054-robustness-20261023
private=/media/scott/SSD_4TB/exp053-private-20261016/exp054-qwen-three-foil-20261023-v1
python=/home/scott/git/.latent-minicpm-v47-venv/bin/python
cd "$repo"
test -d "$private"
test "$(stat -c %a "$private")" = 700
test "$(stat -c %u "$private")" = "$(id -u)"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=''
export HIP_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
# Fail closed if the first 32 previously committed original case receipts do
# not match the exact source code, checkpoint, prompt hashes, and foil roster.
python3 -B research/exp054_qwen_threefoil_audit.py --count 32
for start in 32 40 48 56; do
  total=$((start+8))
  printf 'EXP054_QWEN_START source_case=%d count=8\n' "$start"
  timeout 360s nice -n 15 ionice -c3 "$python" -B \
    research/exp054_qwen_all_foils_cpu.py \
    --start "$start" --count 8 --execute-source-backed
  # Audit and freeze every group before any later model weight loading.
  python3 -B research/exp054_qwen_threefoil_audit.py --count "$total" --write-private
  printf 'EXP054_QWEN_VERIFIED_UP_TO=%d\n' "$total"
done
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -B -m pytest -q -p no:cacheprovider \
  tests/test_exp053*.py tests/test_exp054*.py
echo 'EXP054_QWEN_MACHINE_ONLY_FULL_COHORT_OK articles=64 forwards=256'
