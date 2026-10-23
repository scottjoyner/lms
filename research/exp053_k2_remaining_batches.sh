#!/usr/bin/env bash
# Resume the frozen K2-Horizon source cohort only after cases 0..31 are verified.
# No production serving changes, uploads, model switching or parallel loaders.
set -euo pipefail
root=/home/scott/git/wt-lms-exp053-minicpm-local-20261022
cd "$root"
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Exact-start guard: prevents silently choosing easier source cases.
python3 -B research/exp053_agent_k2_summary.py --count 32 --verify-existing

for start in 32 40 48 56; do
  case_count=$((start+8))
  printf -v summary_file "summary-000-%03d-k2-v1.json" "$((case_count - 1))"
  summary_path="/media/scott/SSD_4TB/exp053-private-20261016/k2-machine-proxy-20261022-v1/$summary_file"
  if [[ -e "$summary_path" || -L "$summary_path" ]]; then
    python3 -B research/exp053_agent_k2_summary.py --count "$case_count" --verify-existing
    echo "EXP053_K2_BATCH_RESUMED_EXISTING cumulative_cases=$case_count"
    continue
  fi
  python3 -B research/exp053_k2_resource_gate.py
  echo "EXP053_K2_BATCH_START index=$start count=8"
  research/exp053_k2_isolated.sh --start "$start" --count 8 --execute-source-backed
  python3 -B research/exp053_agent_k2_summary.py --count "$case_count" --write-private
  echo "EXP053_K2_BATCH_AUDITED cumulative_cases=$case_count"
done

# Full three-model comparison must fail closed until all 192 original
# model/article/arm/rotation records have been verified for each model.
python3 -B research/exp053_agent_three_model_compare.py --write-private
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -B -m pytest -q -p no:cacheprovider tests/test_exp053*.py
echo 'EXP053_ALL_THREE_MACHINE_ONLY_COHORTS_PASS n_models=3 cases_per_model=64'
