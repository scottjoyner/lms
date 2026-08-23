# Fleet Max-Production Sweep — 2026-08-23

Method: `sweep_node.sh` drives `gpu_bench.py` against each node's LM Studio
server (`:1234/v1`) at concurrency 1/2/4/8 (16 where noted), max_tokens 256,
temperature 0, fixed ~300-word technical prompt. Raw JSONL per node in this
directory.

assistx docker stack on x1-370 was stopped for the duration
(`docker compose -p auto-assist stop`; restart with `start`).

## Results

| Node | Hardware | Model | c1 | c4 | c8 | c16 | Ceiling |
|---|---|---|---:|---:|---:|---:|---:|
| x1-370 | RX 9070 XT 32GB | ornith-1.5-35b-a3b-apex-mtp | 57.6 | 141.3 | 153.7 | — | **153.7** |
| macbook-air | Apple Silicon (MLX) | qwen3.5-0.8b-mlx | 8.5 | 25.8 | 34.0 | 46.1 | ~46+ (still scaling) |
| optiplex-9030-aio | CPU only, 7.7G | qwen3.5-0.8b-distilled | 10.7 | 14.8 | 15.0 | — | 15.0 |
| lenovo-ideapad | CPU only, 11G | lfm2.5-1.2b | 5.1 | 8.5 | 8.5 | — | 8.5 |
| destroyer | CPU only, 30G | lfm2-24b-a2b (MoE A2B) | 4.7 | 5.7 | 5.7 | — | 5.7 |
| xwing | Radeon 8060S Strix Halo, 23G | — | BROKEN | — | — | — | — |

All tps = aggregate completion tokens/sec.

## Findings

1. x1-370 delivers **~3× the entire rest of the combined fleet** (~31 tps
   aggregate across all other working nodes).
2. CPU-only nodes plateau at their parallel-slot count (4 slots); aggregate
   gains beyond conc 4 are nil.
3. macbook-air was still scaling at conc 16 — worth a follow-up at 32.
4. destroyer's MoE-A2B choice is right for CPU but memory-bound at 4.7 tps
   single-stream; not viable as an inference producer.

## Broken / degraded

- **xwing**: `ternary-bonsai-27b@?` wedged in PROCESSINGPROMPT (matches its
  0.105 tps ok-rate in fleet-bench-20260821b). Server restart + rebinding to
  0.0.0.0 restored reachability, but every model now returns
  `"Engine protocol predict request failed: fetch failed"` — backend engine
  extension appears corrupted. Needs hands-on investigation (likely reinstall
  of the llama.cpp backend extension). Note: `lms server start` defaults to
  127.0.0.1 bind — remote nodes must use `--bind 0.0.0.0`.
- Off-limits during this sweep per operator: joyner + beelink (recovery work),
  deathstar (RX 480 running recovery jobs).

## Follow-ups

- [ ] Repair xwing backend engine, then bench its 8060S iGPU (expected strong)
- [ ] macbook-air conc 32 probe
- [ ] Apply tuned per-model config (FA on, KV q8_0, MTP off) to destroyer/xwing
      model defaults after repair
- [ ] joyner SSH: tailnet ACL must allow user scott before it can be censused
