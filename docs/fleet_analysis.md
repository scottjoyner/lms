# Fleet Analysis - detailed capabilities & routing

> Generated: 2026-07-15 21:51 UTC
> **Status:** 8 nodes have a `run_summary.csv` from this pass; 1 being re-benchmarked (destroyer, joyner, lenovo); concurrency probe pending. Re-run: `python3 fleet_analysis.py`.

>
> **Data reliability - read first.** The high *Failed (no output)* counts in this pass are almost certainly **artifacts, not real model breakage**: (1) the fleet hit a disk-full condition mid-run (Docker filled the root volume), interrupting several node benchmarks and leaving zero-tps rows; (2) x1-370 was benchmarked *while also orchestrating* the other 8 nodes, heavily contending its own CPU/RAM. The initial pre-crash pass had x1-370's 22 models all producing tokens at 5-13 tok/s. **Treat per-node failure tallies as 'needs a clean solo re-run', not 'model is broken'.** Re-run contended/interrupted nodes solo (`bench_fleet.py --only <node>`) for final figures.

## Fleet overview

| Node | HW | CPU | RAM (GiB) | VRAM (GiB) | Loaded | Chat | Ran | Fail | Med tps | Cap | Status |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| beelink-ryzen-7-mini-pc | V | AMD Ryzen 7 (6-8 cores, o… | 16.0 | - | 9 | 8 | 7 | 1 | 8.6 | 2 | has data |
| deathstar | V | Intel Core i7 (owner-prov… | 24.0 | 8.0 | 20 | 19 | 5 | 14 | 23.9 | 1 | has data |
| destroyer | V | Intel(R) Core(TM) i7-1061… | 31.0 | - | 11 | 10 | 3 | 7 | 4.7 | 1 | has data |
| joyner | V | AMD Ryzen 5 (owner-provid… | 16.0 | 5.0 | 4 | 3 | 1 | 2 | 6.2 | 2 | has data |
| lenovo-ideapad-330s-15ikb | V | Intel(R) Core(TM) i3-8130… | 11.6 | - | 7 | - | - | - | - | 1 | REDO pending |
| scotts-macbook-air | V | Apple Silicon (owner-prov… | 8.0 | - | 4 | 3 | 3 | 0 | 29.0 | 2 | has data |
| scott-optiplex-9030-aio | V | Intel(R) Core(TM) i5-4590… | 7.7 | - | 11 | 10 | 1 | 9 | 8.8 | 1 | has data |
| x1-370 | V | AMD Ryzen AI 9 HX 370 w/ … | 91.9 | - | 22 | 21 | 3 | 18 | 5.2 | 2 | has data |
| xwing | V | AMD RYZEN AI MAX PRO 390 … | 23.2 | - | 6 | 5 | 2 | 3 | 6.7 | 2 | has data |

_HW column: **V** = real per-node hardware (`host_profile.json`); **?** = runner profile, unverified. RAM/CPU are only meaningful where marked V._

## Hardware capture status

- Real per-node hardware collected via `collect_node_profile.py` (run on each host over SSH) for nodes marked **V** above.
- Nodes still on the runner fallback (**?**): SSH key / platform access not yet available. Deploy the runner's SSH key (or run `python3 collect_node_profile.py > runs/<node>/host_profile.json` locally on the node) to upgrade them to **V**.

## Per-machine deep dive

### beelink-ryzen-7-mini-pc

**CPU:** AMD Ryzen 7 (6-8 cores, owner-provided)  
**RAM:** 16.0 GiB (0.0 avail)  
**GPU:** none (CPU only)  
**Endpoint:** `http://100.85.72.121:1234/v1`  
**Models loaded at profile:** 9

_Hardware: real, captured on this host via `collect_node_profile.py`._

- **Chat models benchmarked:** 8 | **Ran:** 7 | **Failed (no output):** 1
- **Median throughput (ran, non-embed):** 8.6 tok/s  |  **Avg eval score:** 0.63
- **Fastest (ran):** liquid/lfm2.5-1.2b (18.6 tok/s, ttft 1 ms), vibethinker-3b-heretic_decensored (13.6 tok/s, ttft 2 ms), vibethinker-3b-hermes (10.4 tok/s, ttft 2 ms), refinedtoolcallv5-3b (8.6 tok/s, ttft 2 ms), google/gemma-4-12b-qat (0.9 tok/s, ttft 284 ms)
- **Slowest (ran):** qwen3.6-28b-reap20-a3b (0.2 tok/s), orinth-1.0-9b (0.7 tok/s), google/gemma-4-12b-qat (0.9 tok/s), refinedtoolcallv5-3b (8.6 tok/s), vibethinker-3b-hermes (10.4 tok/s)
- **Failed (no output / crash):** qwen3.6-14b-a3b-vibeforged-v2 (0.00)
- **Fit grades:** {'good': 8, 'unknown': 1}

**Concurrency posture:** Test at 2 concurrent.

### deathstar

**CPU:** Intel Core i7 (owner-provided)  
**RAM:** 24.0 GiB (0.0 avail)  
**GPU:** AMD Radeon RX 480 8GB  
**Endpoint:** `http://100.78.106.121:1234/v1`  
**Models loaded at profile:** 20

_Hardware: real, captured on this host via `collect_node_profile.py`._

- **Chat models benchmarked:** 19 | **Ran:** 5 | **Failed (no output):** 14
- **Median throughput (ran, non-embed):** 23.9 tok/s  |  **Avg eval score:** 0.35
- **Fastest (ran):** qwen3.5-0.8b-claude-4.6-opus-reasoning-distilled (44.2 tok/s, ttft 4 ms), refinedtoolcallv5-3b (39.8 tok/s, ttft 1 ms), lfm2-1.2b-tool (23.9 tok/s, ttft 4 ms), vibethinker-3b-i1 (16.5 tok/s, ttft 13 ms), qwen3.5-9b-neo-heretic-i1 (1.4 tok/s, ttft 25 ms)
- **Slowest (ran):** qwen3.5-9b-neo-heretic-i1 (1.4 tok/s), vibethinker-3b-i1 (16.5 tok/s), lfm2-1.2b-tool (23.9 tok/s), refinedtoolcallv5-3b (39.8 tok/s), qwen3.5-0.8b-claude-4.6-opus-reasoning-distilled (44.2 tok/s)
- **Failed (no output / crash):** ornith-1.0-9b (0.00); ornith-1.0-35b (0.00); vibethinker-3b-hermes (0.00); qwen3.6-35b-a3b-claude-4.7-opus-reasoning-distilled-apex (0.00); google/gemma-4-12b-qat (0.00); gemma-4-e4b-uncensored-hauhaucs-aggressive (0.00); qwen3.6-12b-iq-ultra-heretic-uncensored-thinking-v2-hightop (0.00); lfm2.5-8b-a1b-terminal-toolbench-full-sft-1epoch (0.00); qwen3.5-4b-neo-heretic-i1 (0.00); qwen3.5-9b-neo (0.00)
- **Fit grades:** {'unknown': 2, 'good': 18}

**Concurrency posture:** Force concurrency 1. CPU maxed by other jobs; cannot absorb parallel load.
**Constraints:** System RAM ~20 GiB BUT only 8 GiB GPU VRAM allocated - models >~7 GiB exceed VRAM and spill to CPU (slow) or fail. CPU also maxed by other jobs.

### destroyer

**CPU:** Intel(R) Core(TM) i7-10610U CPU @ 1.80GHz (8 logical)  
**RAM:** 31.0 GiB (10.0 avail)  
**GPU:** Intel Corporation Comet Lake-U v1 4c Host Bridge/DRAM Controller (rev 0c); Intel Corporation CometLake-U GT2 [UHD Graphics] (rev 02)  
**Endpoint:** `http://destroyer.tailcb8954.ts.net:1234/v1`  
**Models loaded at profile:** 11

_Hardware: real, captured on this host via `collect_node_profile.py`._

- **Chat models benchmarked:** 10 | **Ran:** 3 | **Failed (no output):** 7
- **Median throughput (ran, non-embed):** 4.7 tok/s  |  **Avg eval score:** 0.50
- **Fastest (ran):** lfm2.5-1.2b-instruct (8.1 tok/s, ttft 3 ms), liquid/lfm2-24b-a2b (4.7 tok/s, ttft 10 ms), lfm2.5-8b-a1b (1.6 tok/s, ttft 99 ms)
- **Slowest (ran):** lfm2.5-8b-a1b (1.6 tok/s), liquid/lfm2-24b-a2b (4.7 tok/s), lfm2.5-1.2b-instruct (8.1 tok/s)
- **Failed (no output / crash):** mradermacher/vibethinker-3b-hermes (0.00); refinedneuro/vibethinker-3b-hermes (0.00); vibethinker-3b-i1 (0.00); qwen3.6-35b-a3b-claude-4.7-opus-reasoning-distilled-apex-mtp (0.00); qwen3.6-35b-a3b-claude-4.7-opus-reasoning-distilled-apex (0.00); google/gemma-4-12b-qat (0.00); openai/gpt-oss-20b (0.00)
- **Fit grades:** {'good': 10, 'unknown': 1}

**Concurrency posture:** Chokes under concurrency. Cap at 1.
**Constraints:** Single-stream small models only.

### joyner

**CPU:** AMD Ryzen 5 (owner-provided)  
**RAM:** 16.0 GiB (0.0 avail)  
**GPU:** Ryzen 5 integrated graphics, max 5 GiB VRAM  
**Endpoint:** `http://joyner.tailcb8954.ts.net:1234/v1`  
**Models loaded at profile:** 4

_Hardware: real, captured on this host via `collect_node_profile.py`._

- **Chat models benchmarked:** 3 | **Ran:** 1 | **Failed (no output):** 2
- **Median throughput (ran, non-embed):** 6.2 tok/s  |  **Avg eval score:** 0.68
- **Fastest (ran):** refinedtoolcallv5-3b (6.2 tok/s, ttft 2 ms)
- **Slowest (ran):** refinedtoolcallv5-3b (6.2 tok/s)
- **Failed (no output / crash):** ornith-1.0-9b (0.00); google/gemma-4-12b-qat (0.00)
- **Fit grades:** {'good': 3, 'unknown': 1}

**Concurrency posture:** Test at 2 concurrent.

### lenovo-ideapad-330s-15ikb

**CPU:** Intel(R) Core(TM) i3-8130U CPU @ 2.20GHz (4 logical)  
**RAM:** 11.6 GiB (6.5 avail)  
**GPU:** Intel Corporation Xeon E3-1200 v6/7th Gen Core Processor Host Bridge/DRAM Registers (rev 08); Intel Corporation UHD Graphics 620 (rev 07)  
**Endpoint:** `http://scott-lenovo-ideapad-330s-15ikb.tailcb8954.ts.net:1234/v1`  
**Models loaded at profile:** 7

_Hardware: real, captured on this host via `collect_node_profile.py`._

_No validated benchmark data yet - unreachable during the last pass, being re-benchmarked now._


**Concurrency posture:** Chokes under concurrency. Cap at 1.
**Constraints:** Single-stream small models only.

### scotts-macbook-air

**CPU:** Apple Silicon (owner-provided)  
**RAM:** 8.0 GiB (0.0 avail)  
**GPU:** Apple Silicon GPU (unified memory)  
**Endpoint:** `http://scotts-macbook-air.tailcb8954.ts.net:1234/v1`  
**Models loaded at profile:** 4

_Hardware: real, captured on this host via `collect_node_profile.py`._

- **Chat models benchmarked:** 3 | **Ran:** 3 | **Failed (no output):** 0
- **Median throughput (ran, non-embed):** 29.0 tok/s  |  **Avg eval score:** 0.64
- **Fastest (ran):** liquid/lfm2.5-1.2b (56.7 tok/s, ttft 0 ms), qwen3.5-0.8b-claude-4.6-opus-reasoning-distilled (29.0 tok/s, ttft 6 ms), refinedtoolcallv5-3b (1.7 tok/s, ttft 38 ms)
- **Slowest (ran):** refinedtoolcallv5-3b (1.7 tok/s), qwen3.5-0.8b-claude-4.6-opus-reasoning-distilled (29.0 tok/s), liquid/lfm2.5-1.2b (56.7 tok/s)
- **Fit grades:** {'good': 3, 'unknown': 1}

**Concurrency posture:** Test at 2 concurrent; mind unified-memory pressure.

### scott-optiplex-9030-aio

**CPU:** Intel(R) Core(TM) i5-4590S CPU @ 3.00GHz (4 logical)  
**RAM:** 7.7 GiB (5.8 avail)  
**GPU:** Intel Corporation 4th Gen Core Processor DRAM Controller (rev 06); Intel Corporation Xeon E3-1200 v3/4th Gen Core Processor Integrated Graphics Controller (rev 06)  
**Endpoint:** `http://scott-optiplex-9030-aio.tailcb8954.ts.net:1234/v1`  
**Models loaded at profile:** 11

_Hardware: real, captured on this host via `collect_node_profile.py`._

- **Chat models benchmarked:** 10 | **Ran:** 1 | **Failed (no output):** 9
- **Median throughput (ran, non-embed):** 8.8 tok/s  |  **Avg eval score:** 0.61
- **Fastest (ran):** qwen3.5-0.8b-claude-4.6-opus-reasoning-distilled (8.8 tok/s, ttft 24 ms)
- **Slowest (ran):** qwen3.5-0.8b-claude-4.6-opus-reasoning-distilled (8.8 tok/s)
- **Failed (no output / crash):** refinedtoolcallv5-3b (0.00); vibethinker-3b-i1 (0.00); qwen3.5-2b-claude-4.6-opus-reasoning-distilled (0.00); lfm2.5-8b-a1b (0.00); qwen3.5-4b (0.00); qwen3.5-2b (0.00); ibm/granite-4-h-tiny (0.00); liquid/lfm2.5-1.2b (0.00); google/gemma-3-1b (0.00)
- **Fit grades:** {'good': 9, 'unknown': 2}

**Concurrency posture:** Chokes under concurrency. Cap at 1.
**Constraints:** Single-stream small models only.

### x1-370

**CPU:** AMD Ryzen AI 9 HX 370 w/ Radeon 890M (24 logical)  
**RAM:** 91.9 GiB (53.1 avail)  
**GPU:** Advanced Micro Devices, Inc. [AMD] Device 1507; Advanced Micro Devices, Inc. [AMD] Device 1508  
**Endpoint:** `http://127.0.0.1:1234/v1`  
**Models loaded at profile:** 22

_Hardware: real, captured on this host via `collect_node_profile.py`._

- **Chat models benchmarked:** 21 | **Ran:** 3 | **Failed (no output):** 18
- **Median throughput (ran, non-embed):** 5.2 tok/s  |  **Avg eval score:** 0.26
- **Fastest (ran):** refinedtoolcallv5-3b (16.9 tok/s, ttft 0 ms), minicpm5-1b-agentic-tooluse (5.2 tok/s, ttft 6 ms), qwen3.5-9b-claude-4.6-highiq-instruct-heretic-uncensored (2.2 tok/s, ttft 12 ms)
- **Slowest (ran):** qwen3.5-9b-claude-4.6-highiq-instruct-heretic-uncensored (2.2 tok/s), minicpm5-1b-agentic-tooluse (5.2 tok/s), refinedtoolcallv5-3b (16.9 tok/s)
- **Failed (no output / crash):** vibethinker-3b-hermes (0.00); orinth-1.0-9b (0.00); qwen3.5-0.8b-claude-4.6-opus-reasoning-distilled (0.00); ornith-1.0-35b (0.10); north-mini-code-1.0 (0.00); diffusiongemma-26b-a4b-it-strix-halo (0.00); mradermacher/vibethinker-3b (0.00); prithivmlmods/vibethinker-3b (0.00); qwen3.6-35b-a3b-uncensored-hauhaucs-aggressive (0.00); huihui-qwen3.6-35b-a3b-claude-4.7-opus-abliterated-mtp (0.00)
- **Fit grades:** {'good': 20, 'unknown': 2}

**Concurrency posture:** Strongest concurrency node (96 GiB RAM, many co-resident services). refinedtoolcallv5-3b, orinth-1.0-35b, qwen3.5-0.8b run concurrently decently; per-stream throughput drops to ~2-8 tok/s under concurrent load. Good for multiplexing small models.
**Constraints:** Numbers taken while it was also the orchestrator are depressed; re-run solo for clean figures.

### xwing

**CPU:** AMD RYZEN AI MAX PRO 390 w/ Radeon 8050S (24 logical)  
**RAM:** 23.2 GiB (10.3 avail)  
**GPU:** Advanced Micro Devices, Inc. [AMD] Device 1507 (rev 02); Advanced Micro Devices, Inc. [AMD] Device 1508 (rev 02)  
**Endpoint:** `http://xwing.tailcb8954.ts.net:1234/v1`  
**Models loaded at profile:** 6

_Hardware: real, captured on this host via `collect_node_profile.py`._

- **Chat models benchmarked:** 5 | **Ran:** 2 | **Failed (no output):** 3
- **Median throughput (ran, non-embed):** 6.7 tok/s  |  **Avg eval score:** 0.26
- **Fastest (ran):** vibethinker-3b-hermes (8.7 tok/s, ttft 14 ms), orinth-1.0-9b (4.8 tok/s, ttft 14 ms)
- **Slowest (ran):** orinth-1.0-9b (4.8 tok/s), vibethinker-3b-hermes (8.7 tok/s)
- **Failed (no output / crash):** ornith-1.0-35b (0.00); qwen3.5-0.8b-claude-4.6-opus-reasoning-distilled (0.00); qwen3.6-14b-a3b-vibeforged-v2 (0.00)
- **Fit grades:** {'good': 5, 'unknown': 1}

**Concurrency posture:** Small models benefit from concurrency. Large models (Hermes-class) degrade badly at 2 concurrent sessions. Test at 2, prefer 1 for big models.
**Constraints:** Big-model concurrency is the danger zone.

## Cross-fleet model placement

For each chat model on >1 node, fastest validated home (produced output). Embeddings excluded.

| Model | Available on (tps) | Best home | Fit on best |
|---|---|---|---|
| refinedtoolcallv5-3b | deathstar (40), x1-370 (17), beelink-ryzen-7-mini-pc (9), joyner (6), scotts-macbook-air (2), scott-optiplex-9030-aio (0) | deathstar | 1.89 |
| qwen3.5-0.8b-claude-4.6-opus-reasoning-distilled | deathstar (44), scotts-macbook-air (29), scott-optiplex-9030-aio (9), x1-370 (0), xwing (0) | deathstar | 0.50 |
| google/gemma-4-12b-qat | beelink-ryzen-7-mini-pc (1), deathstar (0), destroyer (0), joyner (0) | beelink-ryzen-7-mini-pc | 7.54 |
| vibethinker-3b-hermes | beelink-ryzen-7-mini-pc (10), xwing (9), deathstar (0), x1-370 (0) | beelink-ryzen-7-mini-pc | 1.89 |
| orinth-1.0-9b | xwing (5), beelink-ryzen-7-mini-pc (1), x1-370 (0) | xwing | 5.66 |
| liquid/lfm2.5-1.2b | scotts-macbook-air (57), beelink-ryzen-7-mini-pc (19), scott-optiplex-9030-aio (0) | scotts-macbook-air | 0.75 |
| ornith-1.0-35b | deathstar (0), x1-370 (0), xwing (0) | _none ran_ | - |
| vibethinker-3b-i1 | deathstar (17), destroyer (0), scott-optiplex-9030-aio (0) | deathstar | 1.89 |
| qwen3.6-35b-a3b-claude-4.7-opus-reasoning-distilled-apex | deathstar (0), destroyer (0), x1-370 (0) | _none ran_ | - |
| lfm2.5-8b-a1b | destroyer (2), deathstar (0), scott-optiplex-9030-aio (0) | destroyer | 5.03 |
| qwen3.6-14b-a3b-vibeforged-v2 | beelink-ryzen-7-mini-pc (0), xwing (0) | _none ran_ | - |
| ornith-1.0-9b | deathstar (0), joyner (0) | _none ran_ | - |
| openai/gpt-oss-20b | deathstar (0), destroyer (0) | _none ran_ | - |

## Per-node capacity (RAM / VRAM vs model memory)

Model memory is intrinsic (from `model_fit.estimated_model_memory_gib`). Effective limit = known VRAM where it is the binding constraint, else system RAM minus ~4 GiB OS headroom. Where VRAM is unknown the RAM figure may be optimistic for GPU-loaded models.

| Node | HW | Eff limit (GiB) | Basis | Models fit | Largest fit | Too-big |
|---|---|---:|---|---:|---|---:|
| beelink-ryzen-7-mini-pc | V | 12.8 | RAM | 33 | qwen3.6-14b-a3b-vibeforged-v2 | 13 |
| deathstar | V | 8.0 | VRAM | 31 | google/gemma-4-12b-qat | 15 |
| destroyer | V | 27.0 | RAM | 46 | ornith-1.0-35b | 0 |
| joyner | V | 5.0 | VRAM | 21 | qwen3.5-4b-claude-4.6-opus-reasoning-distilled-v2 | 25 |
| lenovo-ideapad-330s-15ikb | V | 9.3 | RAM | 33 | qwen3.6-14b-a3b-vibeforged-v2 | 13 |
| scotts-macbook-air | V | 6.4 | RAM | 28 | orinth-1.0-9b | 18 |
| scott-optiplex-9030-aio | V | 6.2 | RAM | 28 | orinth-1.0-9b | 18 |
| x1-370 | V | 87.9 | RAM | 46 | ornith-1.0-35b | 0 |
| xwing | V | 19.2 | RAM | 38 | qwen3.5-27b-claude-4.6-opus-reasoning-distilled | 8 |

- **beelink-ryzen-7-mini-pc** (limit 12.8 GiB, RAM) - too big: ornith-1.0-35b (25.7), qwen3.6-35b-a3b-claude-4.7-opus-reasoning-distilled-apex (25.7), qwen3.6-35b-a3b-claude-4.7-opus-reasoning-distilled-apex-mtp (25.7), huihui-qwen3.6-35b-a3b-claude-4.7-opus-abliterated-mtp (25.7), qwen3.5-35b-a3b-uncensored-hauhaucs-aggressive (25.7), qwen3.6-35b-a3b-uncensored-hauhaucs-aggressive (25.7)
- **deathstar** (limit 8.0 GiB, VRAM) - too big: ornith-1.0-35b (25.7), qwen3.6-35b-a3b-claude-4.7-opus-reasoning-distilled-apex (25.7), qwen3.6-35b-a3b-claude-4.7-opus-reasoning-distilled-apex-mtp (25.7), huihui-qwen3.6-35b-a3b-claude-4.7-opus-abliterated-mtp (25.7), qwen3.5-35b-a3b-uncensored-hauhaucs-aggressive (25.7), qwen3.6-35b-a3b-uncensored-hauhaucs-aggressive (25.7)
- **joyner** (limit 5.0 GiB, VRAM) - too big: ornith-1.0-35b (25.7), qwen3.6-35b-a3b-claude-4.7-opus-reasoning-distilled-apex (25.7), qwen3.6-35b-a3b-claude-4.7-opus-reasoning-distilled-apex-mtp (25.7), huihui-qwen3.6-35b-a3b-claude-4.7-opus-abliterated-mtp (25.7), qwen3.5-35b-a3b-uncensored-hauhaucs-aggressive (25.7), qwen3.6-35b-a3b-uncensored-hauhaucs-aggressive (25.7)
- **lenovo-ideapad-330s-15ikb** (limit 9.3 GiB, RAM) - too big: ornith-1.0-35b (25.7), qwen3.6-35b-a3b-claude-4.7-opus-reasoning-distilled-apex (25.7), qwen3.6-35b-a3b-claude-4.7-opus-reasoning-distilled-apex-mtp (25.7), huihui-qwen3.6-35b-a3b-claude-4.7-opus-abliterated-mtp (25.7), qwen3.5-35b-a3b-uncensored-hauhaucs-aggressive (25.7), qwen3.6-35b-a3b-uncensored-hauhaucs-aggressive (25.7)
- **scotts-macbook-air** (limit 6.4 GiB, RAM) - too big: ornith-1.0-35b (25.7), qwen3.6-35b-a3b-claude-4.7-opus-reasoning-distilled-apex (25.7), qwen3.6-35b-a3b-claude-4.7-opus-reasoning-distilled-apex-mtp (25.7), huihui-qwen3.6-35b-a3b-claude-4.7-opus-abliterated-mtp (25.7), qwen3.5-35b-a3b-uncensored-hauhaucs-aggressive (25.7), qwen3.6-35b-a3b-uncensored-hauhaucs-aggressive (25.7)
- **scott-optiplex-9030-aio** (limit 6.2 GiB, RAM) - too big: ornith-1.0-35b (25.7), qwen3.6-35b-a3b-claude-4.7-opus-reasoning-distilled-apex (25.7), qwen3.6-35b-a3b-claude-4.7-opus-reasoning-distilled-apex-mtp (25.7), huihui-qwen3.6-35b-a3b-claude-4.7-opus-abliterated-mtp (25.7), qwen3.5-35b-a3b-uncensored-hauhaucs-aggressive (25.7), qwen3.6-35b-a3b-uncensored-hauhaucs-aggressive (25.7)
- **xwing** (limit 19.2 GiB, RAM) - too big: ornith-1.0-35b (25.7), qwen3.6-35b-a3b-claude-4.7-opus-reasoning-distilled-apex (25.7), qwen3.6-35b-a3b-claude-4.7-opus-reasoning-distilled-apex-mtp (25.7), huihui-qwen3.6-35b-a3b-claude-4.7-opus-abliterated-mtp (25.7), qwen3.5-35b-a3b-uncensored-hauhaucs-aggressive (25.7), qwen3.6-35b-a3b-uncensored-hauhaucs-aggressive (25.7)

## Recommendations for agents & the orchestrator

### Concurrency

- **Small models (<~4B):** safe to multiplex. x1-370 is the best concurrency host; xwing/joyner/beelink/macbooks tolerate 2 concurrent sessions with acceptable latency.
- **Large models (>=~9B, esp. 30B+ Hermes-class):** keep **single-stream**. Two concurrent sessions to the same big model balloon latency or fail.
- **Cap-1 nodes (optiplex, lenovo, destroyer, deathstar):** never issue parallel requests; mount at most one model, single-stream.
- **deathstar:** also avoid models >7 GiB (CPU maxed by other jobs); if mounted, expect slow/unreliable.

### Loadout

- Mount **small fast tool/agent models** broadly (x1-370, xwing, beelink, macbooks, joyner) for low-latency routing.
- Concentrate **large quality models** on strongest RAM hosts (x1-370 96 GiB; deathstar only if <=7 GiB), single-stream.
- Treat cap-1 weak nodes as **single-model edge servers**.

### Data hygiene

- Track completion by `run_summary.csv`, NOT `capability_matrix.csv` (stale files from crashed runs give false 'done').
- `ok_rate`/`eval_ok_rate` measure QUALITY (cases passed), not availability; availability = model produced output (tps_med > 0).