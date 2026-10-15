#!/usr/bin/env bash
# EXP041B research-only, ONE-SHOT admission launcher; no background daemon.
# Conservative >40 GiB pre-launch headroom is stricter than the frozen
# original >=36 GiB in-Python preload and >=22 GiB per-forward gates.
set -euo pipefail
umask 077
WORKTREE=/home/scott/git/wt-lms-k2-interpolation-pilot-20261015
KNOWLEDGE=/home/scott/git/wt-knowledge-k2-latex-20261014
PYTHON=/home/scott/git/.latent-qwen35-venv/bin/python
RUNNER="$WORKTREE/research/k2_new_source_generation_b.py"
REPORT="$KNOWLEDGE/50-Research/llm-research/latent-world-geometry-2026/EXPERIMENT-041B-RESULTS-2026-10-15.md"
LOCK=/tmp/k2-exp041b-once-20261015.lock

[[ -x "$PYTHON" && -f "$RUNNER" ]] || {
  echo "HOLD_EXP041B_MISSING_RUNNER" >&2; exit 73; }
[[ ! -e "$REPORT" ]] || {
  echo "HOLD_EXP041B_REPORT_EXISTS_NO_OVERWRITE" >&2; exit 73; }
[[ -d "${REPORT%/*}" ]] || {
  echo "HOLD_EXP041B_KNOWLEDGE_DIR_MISSING" >&2; exit 73; }
command -v timeout >/dev/null
command -v flock >/dev/null
command -v nice >/dev/null
command -v ionice >/dev/null
exec 9>"$LOCK"
flock -n 9 || {
  echo "HOLD_EXP041B_ANOTHER_RUN_ACTIVE" >&2; exit 75; }

cd "$WORKTREE"
if [[ "${1:-}" == "--preflight-only" ]]; then
  "$PYTHON" "$RUNNER"
  exit 0
fi
[[ "${1:-}" == "" ]] || {
  echo "Usage: $0 [--preflight-only]" >&2; exit 64; }
available_kib=$(awk '$1=="MemAvailable:"{print $2}' /proc/meminfo)
[[ "$available_kib" =~ ^[0-9]+$ ]] || {
  echo "HOLD_EXP041B_INVALID_MEMINFO" >&2; exit 74; }
# 40 GiB outer margin protects against import/tokenizer RAM fluctuation.
if (( available_kib < 40 * 1024 * 1024 )); then
  printf 'HOLD_EXP041B_RAM %.3f GiB available; require >=40 GiB one-shot margin\n' \
    "$(awk "BEGIN{print $available_kib/1048576}")" >&2
  exit 75
fi
echo "EXP041B_ONE_SHOT_ADMITTED_40GIB_PRECHECK"
exec timeout --signal=TERM --kill-after=10s 650s \
  nice -n 18 ionice -c 3 "$PYTHON" -u "$RUNNER" \
  --execute --out "$REPORT"
