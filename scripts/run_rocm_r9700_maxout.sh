#!/usr/bin/env bash
set -euo pipefail

LLAMA_BUILD=${LLAMA_BUILD:-"$HOME/src/llama.cpp/build-rocm-maxout"}
BIN=${BIN:-"$LLAMA_BUILD/bin"}
RESULTS=${RESULTS:-"$PWD/results/rocm-r9700-maxout-$(date -u +%Y%m%dT%H%M%SZ)"}
PROBE=${PROBE:-"$(cd "$(dirname "$0")" && pwd)/llama_server_probe.py"}
REPS=${REPS:-5}
SERVER_PORT=${SERVER_PORT:-8080}
SERVER_HOST=${SERVER_HOST:-127.0.0.1}
DEEP=${DEEP:-0}
ENABLE_MTP=${ENABLE_MTP:-0}
DRAFT_MODEL=${DRAFT_MODEL:-}

mkdir -p "$RESULTS"/{env,microbench,server,logs}

need_file() { [[ -f "$1" ]] || { echo "missing file: $1" >&2; exit 2; }; }
need_file "$BIN/llama-bench"
need_file "$BIN/llama-server"
need_file "$PROBE"

snapshot() {
  (date -u +%FT%TZ; uname -a; "$BIN/llama-server" --version || true; "$BIN/llama-bench" --list-devices || true) >"$RESULTS/env/runtime.txt" 2>&1
  rocminfo >"$RESULTS/env/rocminfo.txt" 2>&1 || true
  if command -v amd-smi >/dev/null 2>&1; then
    amd-smi static --json >"$RESULTS/env/amd-smi-static.json" 2>&1 || true
    amd-smi metric --json >"$RESULTS/env/amd-smi-metric-before.json" 2>&1 || true
  fi
  env | LC_ALL=C sort | grep -E '^(ROCM|HIP|HSA|GGML|LLAMA|GPU_|OMP_|MALLOC_)' >"$RESULTS/env/relevant-env.txt" || true
}

record_model() {
  local id=$1 model=$2
  local out="$RESULTS/env/${id}.txt"
  {
    echo "id=$id"
    echo "path=$model"
    stat "$model"
    sha256sum "$model"
  } >"$out"
}

microbench() {
  local id=$1 model=$2
  local out="$RESULTS/microbench/${id}.jsonl"
  echo "== microbench $id =="
  "$BIN/llama-bench" \
    -m "$model" \
    -ngl all \
    -fa on \
    -p 512,2048,8192,32768 \
    -n 128,512 \
    -b 512,1024,2048 \
    -ub 256,512,1024 \
    -ctk f16,q8_0,q4_0 \
    -ctv f16,q8_0,q4_0 \
    -r "$REPS" \
    --delay 1 \
    -o jsonl >"$out"
}

wait_ready() {
  local deadline=$((SECONDS + 180))
  while (( SECONDS < deadline )); do
    if curl -fsS "http://${SERVER_HOST}:${SERVER_PORT}/health" >/dev/null 2>&1; then return 0; fi
    if [[ -n "${SERVER_PID:-}" ]] && ! kill -0 "$SERVER_PID" 2>/dev/null; then return 1; fi
    sleep 1
  done
  return 1
}

stop_server() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    for _ in {1..20}; do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep .25; done
    kill -9 "$SERVER_PID" 2>/dev/null || true
  fi
  SERVER_PID=
}
trap stop_server EXIT

server_case() {
  local id=$1 model=$2 ctx=$3 np=$4 batch=$5 ubatch=$6 kv=$7 spec=$8
  local label="${id}-c${ctx}-np${np}-b${batch}-ub${ubatch}-kv${kv}-spec${spec}"
  local log="$RESULTS/logs/${label}.log"
  local result="$RESULTS/server/${label}.json"
  local -a spec_args=()

  case "$spec" in
    none) ;;
    ngram-mod) spec_args=(--spec-type ngram-mod --spec-ngram-mod-n-match 24 --spec-ngram-mod-n-min 48 --spec-ngram-mod-n-max 64) ;;
    draft-simple)
      [[ -n "$DRAFT_MODEL" && -f "$DRAFT_MODEL" ]] || { echo "SKIP $label: DRAFT_MODEL missing"; return 0; }
      spec_args=(--spec-type draft-simple --spec-draft-model "$DRAFT_MODEL" --spec-draft-ngl all --spec-draft-n-max 8 --spec-draft-p-min 0.0)
      ;;
    draft-mtp)
      [[ "$ENABLE_MTP" == 1 ]] || { echo "SKIP $label: ENABLE_MTP!=1"; return 0; }
      spec_args=(--spec-type draft-mtp --spec-draft-n-max 8)
      ;;
    *) echo "unknown spec mode: $spec" >&2; return 2 ;;
  esac

  stop_server
  echo "== server $label =="
  "$BIN/llama-server" \
    -m "$model" --alias "$id" \
    --host "$SERVER_HOST" --port "$SERVER_PORT" \
    -ngl all -fa on \
    -c "$ctx" -np "$np" \
    -b "$batch" -ub "$ubatch" \
    -ctk "$kv" -ctv "$kv" \
    --cont-batching \
    --metrics \
    --perf \
    --jinja \
    "${spec_args[@]}" >"$log" 2>&1 &
  SERVER_PID=$!

  if ! wait_ready; then
    echo "REJECTED $label: server failed readiness" | tee "$result.failed"
    stop_server
    return 0
  fi

  python3 "$PROBE" \
    --endpoint "http://${SERVER_HOST}:${SERVER_PORT}/v1" \
    --model "$id" \
    --concurrency "$np" \
    --requests "$((np * 5))" \
    --prompt-repetitions 256 \
    --max-tokens 512 \
    --label "$label" \
    --output "$result" || true

  curl -fsS "http://${SERVER_HOST}:${SERVER_PORT}/metrics" >"$RESULTS/server/${label}.prom" 2>/dev/null || true
  if command -v amd-smi >/dev/null 2>&1; then
    amd-smi metric --json >"$RESULTS/server/${label}.amd-smi.json" 2>&1 || true
  fi
  stop_server
}

run_model() {
  local id=$1 model=$2
  [[ -n "$model" && -f "$model" ]] || { echo "SKIP $id: model artifact not present"; return 0; }
  record_model "$id" "$model"
  microbench "$id" "$model"

  local -a contexts=(8192 32768 65536 131072)
  [[ "$DEEP" == 1 ]] && contexts+=(262144)
  local -a slots=(1 2)
  [[ "$DEEP" == 1 ]] && slots+=(4)

  for ctx in "${contexts[@]}"; do
    for np in "${slots[@]}"; do
      # Start with the highest-throughput practical batching pair. Deeper sweeps
      # can override BATCH/UBATCH or use the canonical JSON matrix.
      for kv in f16 q8_0 q4_0; do
        server_case "$id" "$model" "$ctx" "$np" 2048 512 "$kv" none
      done
    done
  done

  # Speculation is isolated from baseline so speedups cannot hide regressions.
  server_case "$id" "$model" 32768 1 2048 512 q8_0 ngram-mod
  server_case "$id" "$model" 32768 1 2048 512 q8_0 draft-simple
  server_case "$id" "$model" 32768 1 2048 512 q8_0 draft-mtp
}

snapshot
run_model ornith-1.0-35b "${ORNITH_MODEL:-}"
run_model qwen3.8-27b "${QWEN38_MODEL:-}"

if command -v amd-smi >/dev/null 2>&1; then
  amd-smi metric --json >"$RESULTS/env/amd-smi-metric-after.json" 2>&1 || true
fi
printf '%s\n' "$RESULTS" >"$RESULTS/RESULT_DIR.txt"
echo "benchmark evidence: $RESULTS"
