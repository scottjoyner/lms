#!/usr/bin/env bash
set -euo pipefail

QUEUE=${QUEUE:-"$(cd "$(dirname "$0")/.." && pwd)/benchmarks/offline_batch_jobs.tsv"}
STATE_ROOT=${STATE_ROOT:-"$PWD/results/offline-batch-state"}
HOST=${HOST:-$(hostname -s)}
DRY_RUN=${DRY_RUN:-1}
mkdir -p "$STATE_ROOT"/{done,failed,logs}

match_host(){
  local preferred=$1
  [[ -z "$preferred" || "$preferred" == "$HOST" || "$preferred" == "any" ]]
}

run_job(){
  local job_id=$1 job_type=$2 input_env=$3
  local log="$STATE_ROOT/logs/${job_id}.log"
  case "$job_type" in
    model-inventory)
      : "${NAS_ROOT:?set NAS_ROOT}"
      find "$NAS_ROOT" -maxdepth 4 -type f \( -name '*.gguf' -o -name '*.safetensors' \) -printf '%s\t%p\n' | sort -nr >"$STATE_ROOT/${job_id}.tsv"
      ;;
    gguf-metadata)
      : "${NAS_ROOT:?set NAS_ROOT}"
      : "${LLAMA_BUILD:?set LLAMA_BUILD}"
      while IFS=$'\t' read -r _ path; do
        [[ "$path" == *.gguf ]] || continue
        printf '### %s\n' "$path"
        "$LLAMA_BUILD/bin/llama-gguf" "$path" 2>/dev/null || true
      done <"$STATE_ROOT/inventory-models.tsv" >"$STATE_ROOT/${job_id}.txt"
      ;;
    storage-preflight)
      bash "$(dirname "$0")/benchmark_storage_preflight.sh"
      ;;
    calibration-corpus)
      echo "Calibration corpus generation requires project-specific sources; record manifest under $STATE_ROOT/${job_id}.manifest" >"$STATE_ROOT/${job_id}.txt"
      ;;
    result-package)
      : "${RESULT_ROOT:?set RESULT_ROOT}"
      tar --zstd -cf "$STATE_ROOT/${job_id}-$(date -u +%Y%m%dT%H%M%SZ).tar.zst" "$RESULT_ROOT"
      ;;
    result-analysis)
      : "${RESULT_ROOT:?set RESULT_ROOT}"
      find "$RESULT_ROOT" -type f \( -name '*.json' -o -name '*.jsonl' -o -name '*.prom' \) -printf '%s\t%p\n' | sort -nr >"$STATE_ROOT/${job_id}.files.tsv"
      ;;
    quantize|hf-to-gguf|activations|pruning-rank|quality-eval)
      echo "Job type $job_type requires model/tool-specific command injection. See docs/OFFLINE_BATCH_WORKFLOW.md." >"$STATE_ROOT/${job_id}.txt"
      ;;
    *) echo "unknown job type: $job_type" >&2; return 2;;
  esac >"$log" 2>&1
}

header=1
while IFS=$'\t' read -r job_id stage job_type input_env preferred_host resource_class depends_on output_class status notes; do
  if ((header)); then header=0; continue; fi
  [[ -n "$job_id" ]] || continue
  [[ -e "$STATE_ROOT/done/$job_id" ]] && continue
  match_host "$preferred_host" || continue
  if [[ -n "$depends_on" && ! -e "$STATE_ROOT/done/$depends_on" ]]; then continue; fi
  if [[ "$DRY_RUN" == 1 ]]; then
    printf 'READY\t%s\t%s\t%s\t%s\n' "$job_id" "$job_type" "$resource_class" "$notes"
    continue
  fi
  if run_job "$job_id" "$job_type" "$input_env"; then
    date -u +%FT%TZ >"$STATE_ROOT/done/$job_id"
  else
    date -u +%FT%TZ >"$STATE_ROOT/failed/$job_id"
  fi
done <"$QUEUE"
