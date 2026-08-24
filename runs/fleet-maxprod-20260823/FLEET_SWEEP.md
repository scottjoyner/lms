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
| xwing | Radeon 8060S Strix Halo, 23G | nanbeige_nanbeige4.2-3b | 5.5 | 15.1 | fail | — | ~15 (contested) |

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

- **xwing — root-caused and partially repaired 2026-08-23.** Failure chain:
  1. LM Studio auto-selected backend `vulkan-avx2 2.29.1` (first seen in
     journal that day); every llama-server spawn SIGABRTs before healthy on
     the Radeon 8050S (gfx1151) → "Engine protocol runtime exited before
     becoming healthy" / "fetch failed" for all models.
  2. Box was simultaneously OOM-thrashing: default model contexts (262k) on
     unified memory blew past 23 GiB RAM; kernel OOM-killed the daemon,
     openclaw-gateway, hermes-gateway; 10–37 GB swap churn.
     Kernel log: `Out of memory: Killed process ... (llama-server)`.
  3. Fix applied: rolled
     `~/.lmstudio/.internal/backend-preferences-v1.json` from vulkan-2.29.1
     to **vulkan-avx2 2.28.2** (backup: `.bak-vulkan2291`; ROCm 2.28.2 also
     aborts on gfx1151), restarted server with `--bind 0.0.0.0`, and loads
     must use explicit small context (`-c 8192`).
  - Result: serving again, but numbers are contested — other jobs (finetune
    staging llama-servers, gateways) run on this box per operator.
    nanbeige c1/c4 = 5.5/15.1 tps; conc 8 rejected.
  - Follow-ups: keep backend pinned to vulkan 2.28.2 (do not auto-update),
    set sane default contexts per model, consider trimming co-tenant jobs or
    adding memory before trusting perf numbers.
- Off-limits during this sweep per operator: joyner + beelink (recovery work),
  deathstar (RX 480 running recovery jobs).

## Follow-ups

- [x] xwing backend repaired (vulkan 2.28.2 rollback); numbers contested by co-tenant jobs
- [ ] macbook-air conc 32 probe — done: 65.3 tps @ c32, still scaling
- [x] x1-370 parallel sweep: parallel 16 → **180.4 tps** ceiling (compute-bound);
      persisted numParallelSessions 16; parallel 32 tested equal (~181)
- [x] x1-370 tuned config re-verified after contention source removed
- [ ] Apply tuned per-model config (FA on, KV q8_0, MTP off, sane contexts) to
      destroyer/xwing model defaults
- [ ] joyner SSH: tailnet ACL must allow user scott before it can be censused

## macbook-air re-bench after operator raised sessions 4→16 (2026-08-24 late)

| conc | aggregate tps |
|---:|---:|
| 1 | 8.7 |
| 8 | 52.4 |
| 16 | 35.1 (noisy — concurrent image pipeline on device) |
| 32 | 67.2–80.7 |
| 64 | HTTP 500 (server-side limit) |

Ceiling ~81 tps; mid-range concurrency materially improved vs 4-slot config.
Note: dashcam/Nextcloud image-processing jobs run concurrently on this node —
numbers are contested but directionally better. LAN IP moved to 192.168.1.233.
