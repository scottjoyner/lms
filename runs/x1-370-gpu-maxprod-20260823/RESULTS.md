# x1-370 RX 9070 XT (32 GB) — Max Production Benchmark

- Date: 2026-08-23
- Host: x1-370 (AMD Ryzen AI 9 HX 370, 24 threads, 89.7 GiB RAM)
- GPU: AMD Navi 48 [1002:7551], 31.9 GiB VRAM (card1)
- Runtime: LM Studio 0.3.x, llama.cpp ROCm backend (`llama.cpp-linux-x86_64-amd-rocm-avx2-2.29.1`)
- Server: `http://127.0.0.1:1234/v1` (LM Studio main server, no auth locally)

## Method

`gpu_bench.py` in this directory issues `/chat/completions` requests at
concurrency levels 1/2/4/8/16 against the loaded model and reports aggregate
and per-stream tokens/sec from server usage counts. Prompt: ~300-word
technical explanation request, `max_tokens` 256–512, temperature 0.
GPU busy sampled from `/sys/class/drm/card1/device/gpu_busy_percent`.

## Results

### Baseline (pre-existing state after crash recovery)

Qwen3.8-27B-NVFP4-MTP-HIGH on direct ROCm port, flash-attn ON,
parallel 4, f16 KV cache:

| conc | aggregate tps | notes |
|-----:|--------------:|-------|
| 1 | 23.4 | |
| 2 | 45.6 | near-linear |
| 4 | 71.5 | GPU 97–98% busy — compute saturated |
| 8 | 73.4 | plateau; parallel=4 queueing |

### Ornith-1.5-35B-A3B-APEX-MTP sweep (MoE, 256×2.6B active ≈ A3B)

All loads full GPU offload, ctx 65536 unless noted.

| config | conc 1 | conc 4 | conc 8 | conc 16 |
|---|---:|---:|---:|---:|
| MTP draft ON, FA off, f16 KV, parallel 4 | 61.7* | 107.7* | timeout | — |
| MTP draft ON, FA off, parallel 8 | — | 76.9 | wedge | wedge |
| parallel 8 @ ctx 32768 (VRAM pressure) | 44.7 | 82.5 | HTTP 400 | — |
| MTP ON, **FA on**, KV q8_0/q8_0, parallel 8 | 44.1 | 85.8 | 88.5 | 84.8 |
| **MTP OFF**, FA on, KV q8_0/q8_0, parallel 8 | 66.0 | 123–136 | 138–143 | 128.7 |
| Final persisted config (repeat run) | 57.6 | **141.3** | **153.7** | — |

\* first-load readings taken under heavy CPU contention; treat as lower bound.

## Key findings

1. **MTP speculative decoding hurts throughput on this GPU** by ~40% at all
   concurrency levels (88 → 143 tps at conc 8). The MTP draft runs on the same
   saturated GPU; acceptance rate does not pay for the extra compute under
   concurrent load. Single-stream is also faster without it (66 vs 44 tps).
2. **Flash attention must be enabled per-model** via
   `~/.lmstudio/.internal/user-concrete-model-default-config/`. CLI `lms load`
   has no flag; defaults are otherwise FA=off.
3. **VRAM oversubscription wedges the stack**: loading with parallel 8 ×
   65k f16 KV pushed VRAM to 30.5/31.9 GiB, then requests returned HTTP 400 /
   hung / llama-server spun at 100% CPU with a stuck kernel dispatch. KV q8_0
   quantization + unload/reload recovered it without a reboot.
4. Dense 27B saturates compute at ~73 tps ceiling; the A3B MoE reaches
   ~154 tps — **2.1× fleet-node production gain** with higher quality model.

## Final production config (persisted as model default)

`Ornith-1.5-35B-A3B-APEX-MTP-Quality.gguf.json` sets:
contextLength 65536, offloadRatio 1, numParallelSessions 8, flashAttention true,
K/V cache q8_0, cpuThreadPoolSize 12, speculativeDecoding.draftMtp false.

Verified effective llama-server flags:
`--ctx-size 65536 --threads 12 --parallel 8 --cache-type-k q8_0 --cache-type-v q8_0 --flash-attn on`
(no `--spec-type` flag = MTP disabled).

Load with: `lms load ornith-1.5-35b-a3b-apex-mtp --gpu max`

## Caveats

- Non-admitted evidence: single node, short prompts, no long-context sweeps,
  no quality scoring. Numbers vary ±10% run-to-run with desktop/Neo4j load.
- conc 16 queues behind 8 slots; no further aggregate gain.
- Remaining headroom idea: test gpt-oss-20b and non-MTP Ornith variants;
  consider KV q4_0(K) only if longer contexts are needed.
