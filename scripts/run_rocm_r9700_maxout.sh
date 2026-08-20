#!/usr/bin/env bash
set -euo pipefail

LLAMA_BUILD=${LLAMA_BUILD:-"$HOME/src/llama.cpp/build-rocm-maxout"}
BIN=${BIN:-"$LLAMA_BUILD/bin"}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
RESULTS=${RESULTS:-"$PWD/results/rocm-r9700-maxout-$(date -u +%Y%m%dT%H%M%SZ)"}
PROBE=${PROBE:-"$ROOT/scripts/llama_server_probe.py"}
TELEMETRY=${TELEMETRY:-"$ROOT/scripts/sample_rocm_telemetry.sh"}
TELEMETRY_ANALYZER=${TELEMETRY_ANALYZER:-"$ROOT/scripts/analyze_rocm_telemetry.py"}
MODEL_REGISTRY=${MODEL_REGISTRY:-"$ROOT/benchmarks/rocm_r9700_models.tsv"}
REPS=${REPS:-5}
SERVER_PORT=${SERVER_PORT:-8080}
SERVER_HOST=${SERVER_HOST:-127.0.0.1}
DEEP=${DEEP:-0}
SOAK=${SOAK:-0}
SOAK_REQUEST_MULTIPLIER=${SOAK_REQUEST_MULTIPLIER:-20}
ENABLE_MTP=${ENABLE_MTP:-0}
DRAFT_MODEL=${DRAFT_MODEL:-}
TELEMETRY_INTERVAL=${TELEMETRY_INTERVAL:-1}

mkdir -p "$RESULTS"/{env,microbench,server,logs,telemetry,soak}
need_file(){ [[ -f "$1" ]] || { echo "missing file: $1" >&2; exit 2; }; }
for f in "$BIN/llama-bench" "$BIN/llama-server" "$PROBE" "$TELEMETRY" "$TELEMETRY_ANALYZER" "$MODEL_REGISTRY"; do need_file "$f"; done

snapshot(){
  (date -u +%FT%TZ; uname -a; "$BIN/llama-server" --version || true; "$BIN/llama-bench" --list-devices || true) >"$RESULTS/env/runtime.txt" 2>&1
  rocminfo >"$RESULTS/env/rocminfo.txt" 2>&1 || true
  cp "$MODEL_REGISTRY" "$RESULTS/env/model-registry.tsv"
  command -v amd-smi >/dev/null 2>&1 && amd-smi static --json >"$RESULTS/env/amd-smi-static.json" 2>&1 || true
  command -v amd-smi >/dev/null 2>&1 && amd-smi metric --json >"$RESULTS/env/amd-smi-metric-before.json" 2>&1 || true
  env | LC_ALL=C sort | grep -E '^(ROCM|HIP|HSA|GGML|LLAMA|GPU_|OMP_|MALLOC_|ORNITH|QWEN|MODEL_|SOAK|TELEMETRY)' >"$RESULTS/env/relevant-env.txt" || true
}

record_model(){
  local id=$1 family=$2 generation=$3 variant=$4 quant=$5 mtp=$6 provenance=$7 model=$8
  { echo "id=$id"; echo "family=$family"; echo "generation=$generation"; echo "variant=$variant"; echo "declared_quant=$quant"; echo "mtp=$mtp"; echo "provenance=$provenance"; echo "path=$model"; stat "$model"; sha256sum "$model"; } >"$RESULTS/env/${id}.txt"
}

microbench(){
  local id=$1 model=$2
  "$BIN/llama-bench" -m "$model" -ngl all -fa on -p 512,2048,8192,32768 -n 128,512 -b 512,1024,2048 -ub 256,512,1024 -ctk f16,q8_0,q4_0 -ctv f16,q8_0,q4_0 -r "$REPS" --delay 1 -o jsonl >"$RESULTS/microbench/${id}.jsonl"
}

wait_ready(){ local deadline=$((SECONDS+180)); while ((SECONDS<deadline)); do curl -fsS "http://${SERVER_HOST}:${SERVER_PORT}/health" >/dev/null 2>&1 && return 0; [[ -n "${SERVER_PID:-}" ]] && ! kill -0 "$SERVER_PID" 2>/dev/null && return 1; sleep 1; done; return 1; }
stop_telemetry(){ [[ -n "${TELEMETRY_PID:-}" ]] && kill "$TELEMETRY_PID" 2>/dev/null || true; TELEMETRY_PID=; }
stop_server(){ stop_telemetry; if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then kill "$SERVER_PID" 2>/dev/null || true; for _ in {1..20}; do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep .25; done; kill -9 "$SERVER_PID" 2>/dev/null || true; fi; SERVER_PID=; }
trap stop_server EXIT

start_telemetry(){ local label=$1; "$TELEMETRY" "$RESULTS/telemetry/${label}.jsonl" "$SERVER_PID" "$TELEMETRY_INTERVAL" & TELEMETRY_PID=$!; }

probe_case(){
  local id=$1 np=$2 reps=$3 maxtok=$4 cache=$5 label=$6 outfile=$7
  python3 "$PROBE" --endpoint "http://${SERVER_HOST}:${SERVER_PORT}/v1" --model "$id" --concurrency "$np" --requests "$reps" --prompt-repetitions "$maxtok" --max-tokens 512 --cache-mode "$cache" --label "$label" --output "$outfile" || true
}

server_case(){
  local id=$1 model=$2 ctx=$3 np=$4 batch=$5 ubatch=$6 kv=$7 spec=$8 draft_n=${9:-8}
  local label="${id}-c${ctx}-np${np}-b${batch}-ub${ubatch}-kv${kv}-spec${spec}-dn${draft_n}"
  local log="$RESULTS/logs/${label}.log" warm="$RESULTS/server/${label}-warm.json" cold="$RESULTS/server/${label}-cold.json"
  local -a spec_args=()
  case "$spec" in
    none) ;;
    ngram-mod) spec_args=(--spec-type ngram-mod --spec-ngram-mod-n-match 24 --spec-ngram-mod-n-min 48 --spec-ngram-mod-n-max 64);;
    draft-simple) [[ -n "$DRAFT_MODEL" && -f "$DRAFT_MODEL" ]] || { echo "SKIP $label: DRAFT_MODEL missing"; return 0; }; spec_args=(--spec-type draft-simple --spec-draft-model "$DRAFT_MODEL" --spec-draft-ngl all --spec-draft-n-max "$draft_n" --spec-draft-p-min 0.0);;
    draft-mtp) [[ "$ENABLE_MTP" == 1 ]] || { echo "SKIP $label: ENABLE_MTP!=1"; return 0; }; spec_args=(--spec-type draft-mtp --spec-draft-n-max "$draft_n");;
    *) echo "unknown spec mode $spec" >&2; return 2;;
  esac
  stop_server
  "$BIN/llama-server" -m "$model" --alias "$id" --host "$SERVER_HOST" --port "$SERVER_PORT" -ngl all -fa on -c "$ctx" -np "$np" -b "$batch" -ub "$ubatch" -ctk "$kv" -ctv "$kv" --cont-batching --metrics --perf --jinja "${spec_args[@]}" >"$log" 2>&1 & SERVER_PID=$!
  wait_ready || { echo "REJECTED $label" >"$RESULTS/server/${label}.failed"; stop_server; return 0; }
  start_telemetry "$label"
  probe_case "$id" "$np" "$((np*5))" 256 warm "$label-warm" "$warm"
  probe_case "$id" "$np" "$((np*5))" 256 cold "$label-cold" "$cold"
  curl -fsS "http://${SERVER_HOST}:${SERVER_PORT}/metrics" >"$RESULTS/server/${label}.prom" 2>/dev/null || true
  stop_telemetry
  python3 "$TELEMETRY_ANALYZER" "$RESULTS/telemetry/${label}.jsonl" --output "$RESULTS/telemetry/${label}.summary.json" || true
  stop_server
}

soak_case(){
  local id=$1 model=$2 ctx=$3 np=$4 kv=$5
  local label="${id}-soak-c${ctx}-np${np}-kv${kv}" log="$RESULTS/logs/${label}.log" out="$RESULTS/soak/${label}.json"
  stop_server
  "$BIN/llama-server" -m "$model" --alias "$id" --host "$SERVER_HOST" --port "$SERVER_PORT" -ngl all -fa on -c "$ctx" -np "$np" -b 2048 -ub 512 -ctk "$kv" -ctv "$kv" --cont-batching --metrics --perf --jinja >"$log" 2>&1 & SERVER_PID=$!
  wait_ready || { echo "REJECTED $label" >"$out.failed"; stop_server; return 0; }
  start_telemetry "$label"
  python3 "$PROBE" --endpoint "http://${SERVER_HOST}:${SERVER_PORT}/v1" --model "$id" --concurrency "$np" --requests "$((np*SOAK_REQUEST_MULTIPLIER))" --prompt-repetitions 512 --max-tokens 512 --cache-mode cold --label "$label" --output "$out" || true
  curl -fsS "http://${SERVER_HOST}:${SERVER_PORT}/metrics" >"$RESULTS/soak/${label}.prom" 2>/dev/null || true
  stop_telemetry
  tokens=$(python3 - "$out" <<'PY'
import json,sys
try: print(json.load(open(sys.argv[1])).get('output_tokens',0))
except: print(0)
PY
)
  python3 "$TELEMETRY_ANALYZER" "$RESULTS/telemetry/${label}.jsonl" --tokens "$tokens" --output "$RESULTS/soak/${label}.telemetry-summary.json" || true
  stop_server
}

run_model(){
  local id=$1 family=$2 generation=$3 variant=$4 quant=$5 mtp=$6 provenance=$7 model=$8
  record_model "$id" "$family" "$generation" "$variant" "$quant" "$mtp" "$provenance" "$model"; microbench "$id" "$model"
  local -a contexts=(8192 32768 65536 131072); [[ "$DEEP" == 1 ]] && contexts+=(262144)
  local -a slots=(1 2); [[ "$DEEP" == 1 ]] && slots+=(4)
  for ctx in "${contexts[@]}"; do for np in "${slots[@]}"; do for kv in f16 q8_0 q4_0; do server_case "$id" "$model" "$ctx" "$np" 2048 512 "$kv" none; done; done; done
  server_case "$id" "$model" 32768 1 2048 512 q8_0 ngram-mod
  if [[ -n "$DRAFT_MODEL" ]]; then for dn in 2 4 8 16; do server_case "$id" "$model" 32768 1 2048 512 q8_0 draft-simple "$dn"; done; fi
  if [[ "$mtp" != false && "$ENABLE_MTP" == 1 ]]; then for dn in 2 4 8 16; do server_case "$id" "$model" 32768 1 2048 512 q8_0 draft-mtp "$dn"; done; fi
  [[ "$SOAK" == 1 ]] && soak_case "$id" "$model" 32768 1 q8_0
  [[ "$SOAK" == 1 ]] && soak_case "$id" "$model" 32768 2 q8_0
  [[ "$SOAK" == 1 && "$DEEP" == 1 ]] && soak_case "$id" "$model" 65536 4 q8_0
}

run_registry(){
  local header=1 id family generation variant quant model_env mtp required provenance model
  while IFS=$'\t' read -r id family generation variant quant model_env mtp required provenance; do
    if ((header)); then header=0; continue; fi; [[ -n "$id" ]] || continue; model=${!model_env-}
    if [[ -z "$model" || ! -f "$model" ]]; then [[ "$required" == true ]] && echo "REQUIRED MODEL MISSING: $id env=$model_env" >&2 || echo "DEFER $id: set $model_env to verified artifact"; continue; fi
    run_model "$id" "$family" "$generation" "$variant" "$quant" "$mtp" "$provenance" "$model"
  done <"$MODEL_REGISTRY"
}

if [[ -n "${ORNITH_MODEL:-}" && -z "${ORNITH10_35B_MODEL:-}" ]]; then export ORNITH10_35B_MODEL="$ORNITH_MODEL"; fi
snapshot; run_registry
command -v amd-smi >/dev/null 2>&1 && amd-smi metric --json >"$RESULTS/env/amd-smi-metric-after.json" 2>&1 || true
printf '%s\n' "$RESULTS" >"$RESULTS/RESULT_DIR.txt"; echo "benchmark evidence: $RESULTS"
