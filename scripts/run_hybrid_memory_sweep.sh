#!/usr/bin/env bash
set -euo pipefail

MODEL=${MODEL:?set MODEL=/absolute/model.gguf}
MODEL_ID=${MODEL_ID:-hybrid-model}
LLAMA_BUILD=${LLAMA_BUILD:-"$HOME/src/llama.cpp/build-rocm-maxout"}
BIN="$LLAMA_BUILD/bin"
EGPU_INDEX=${EGPU_INDEX:-0}
IGPU_INDEX=${IGPU_INDEX:-1}
CTX=${CTX:-32768}
KV=${KV:-q8_0}
PORT_BASE=${PORT_BASE:-8180}
OUT=${OUT:-"$PWD/results/hybrid-memory-$(date -u +%Y%m%dT%H%M%SZ)"}
PROBE=${PROBE:-"$(cd "$(dirname "$0")" && pwd)/llama_server_probe.py"}
TOPOLOGY=${TOPOLOGY:-"$(cd "$(dirname "$0")" && pwd)/collect_hybrid_memory_topology.sh"}
TELEMETRY=${TELEMETRY:-"$(cd "$(dirname "$0")" && pwd)/sample_rocm_telemetry.sh"}
EXPERIMENTAL_SPLITS=${EXPERIMENTAL_SPLITS:-0}
LAYER_SPLITS=${LAYER_SPLITS:-"1,3 1,2 1,1 2,1 3,1"}
PARTIAL_LAYERS=${PARTIAL_LAYERS:-"8 12 16 20 24 auto"}

mkdir -p "$OUT"/{topology,cases,logs,telemetry}
"$TOPOLOGY" "$OUT/topology" >/dev/null || true
sha256sum "$MODEL" >"$OUT/model.sha256"
stat "$MODEL" >"$OUT/model.stat"
"$BIN/llama-bench" --list-devices >"$OUT/llama-devices.txt" 2>&1 || true

SERVER_PID=
TEL_PID=
cleanup(){ [[ -n "$TEL_PID" ]] && kill "$TEL_PID" 2>/dev/null || true; TEL_PID=; [[ -n "$SERVER_PID" ]] && kill "$SERVER_PID" 2>/dev/null || true; SERVER_PID=; }
trap cleanup EXIT

run_case(){
  local label=$1 port=$2; shift 2
  local log="$OUT/logs/${label}.log" result="$OUT/cases/${label}.json"
  cleanup
  printf '%q ' "$BIN/llama-server" -m "$MODEL" --alias "$MODEL_ID" "$@" >"$OUT/cases/${label}.command.txt"; printf '\n' >>"$OUT/cases/${label}.command.txt"
  "$BIN/llama-server" -m "$MODEL" --alias "$MODEL_ID" --host 127.0.0.1 --port "$port" -c "$CTX" -ctk "$KV" -ctv "$KV" -b 2048 -ub 512 -fa on --metrics --perf --jinja "$@" >"$log" 2>&1 & SERVER_PID=$!
  for _ in {1..180}; do
    curl -fsS "http://127.0.0.1:${port}/health" >/dev/null 2>&1 && break
    kill -0 "$SERVER_PID" 2>/dev/null || { echo rejected >"${result}.failed"; cleanup; return 0; }
    sleep 1
  done
  curl -fsS "http://127.0.0.1:${port}/health" >/dev/null 2>&1 || { echo timeout >"${result}.failed"; cleanup; return 0; }
  "$TELEMETRY" "$OUT/telemetry/${label}.jsonl" "$SERVER_PID" 1 & TEL_PID=$!
  python3 "$PROBE" --endpoint "http://127.0.0.1:${port}/v1" --model "$MODEL_ID" --concurrency 1 --requests 5 --prompt-repetitions 256 --max-tokens 512 --cache-mode cold --label "$label" --output "$result" || true
  curl -fsS "http://127.0.0.1:${port}/metrics" >"$OUT/cases/${label}.prom" 2>/dev/null || true
  cleanup
}

port=$PORT_BASE

# A: Discrete GPU only, letting llama.cpp fit as much as possible in VRAM and
# leave the remainder host-resident. This directly measures the host<->OCuLink
# penalty for models larger than dGPU VRAM.
for ngl in $PARTIAL_LAYERS; do
  run_case "egpu-only-ngl-${ngl}" "$port" -sm none -mg "$EGPU_INDEX" -ngl "$ngl" --fit on --fit-target 2048
  port=$((port+1))
done

# B: APU/iGPU only. This only works when llama.cpp exposes the integrated GPU as
# a selectable HIP device. The observed usable UMA/GTT, not BIOS carve-out, is
# the relevant limit.
run_case "igpu-uma-only" "$port" -sm none -mg "$IGPU_INDEX" -ngl auto --fit on --fit-target 4096
port=$((port+1))

# C: Layer split between dGPU and APU/iGPU. The tensor split ratios are empirical
# candidates, not assumptions about physical capacity. Preserve which device
# index corresponds to each GPU in topology evidence.
for split in $LAYER_SPLITS; do
  tag=${split//,/-}
  run_case "hybrid-layer-ts-${tag}" "$port" -sm layer -ts "$split" -ngl all --fit on --fit-target 2048,4096
  port=$((port+1))
done

# D: Experimental cross-device row/tensor splits. These can increase cross-link
# traffic and have current mixed-device stability caveats; keep opt-in.
if [[ "$EXPERIMENTAL_SPLITS" == 1 ]]; then
  for mode in row tensor; do
    for split in $LAYER_SPLITS; do
      tag=${split//,/-}
      run_case "hybrid-${mode}-ts-${tag}" "$port" -sm "$mode" -ts "$split" -mg "$EGPU_INDEX" -ngl all --fit on --fit-target 2048,4096
      port=$((port+1))
    done
  done
fi

# E: mmap/direct-I/O control for very large UMA models. If the current llama.cpp
# build supports -dio, test it explicitly; unsupported startup is evidence.
run_case "hybrid-layer-direct-io" "$port" -sm layer -ts 1,2 -ngl all --fit on --fit-target 2048,4096 -dio

echo "$OUT"
