# HX-370 + 96 GB Host Memory + OCuLink dGPU: Laguna S 2.1

This experiment targets a hybrid host where the mini-PC has 96 GB system memory, a small BIOS-reserved iGPU VRAM aperture, and a discrete AMD GPU attached over OCuLink.

## Why this topology is different

Do not treat the machine as a flat sum of system RAM plus dGPU VRAM. The relevant quantities are:

- dGPU local VRAM capacity and bandwidth;
- APU/iGPU-accessible UMA/GTT capacity and bandwidth;
- ordinary CPU-visible system memory;
- negotiated OCuLink/PCIe generation and lane width;
- how llama.cpp places model weights, expert weights, KV cache and compute buffers;
- how often data crosses the OCuLink boundary during prompt processing and decode.

A 2 GiB BIOS carve-out is not by itself the iGPU compute-memory ceiling. Preserve `rocminfo`, AMD-SMI, llama.cpp device inventory and observed allocations.

## Laguna S 2.1 target

Laguna S 2.1 is a 118B-total / ~8B-active MoE coding model. The practical Q4-class artifact is roughly 70-75 GiB of weights, making it a useful test of whether the 96 GB host can hold the expert bank while the discrete GPU accelerates the high-value path.

Prepared registry variables:

```text
LAGUNA_S21_Q4_MODEL
LAGUNA_S21_ALT_MODEL
```

Always hash and identify the exact local artifact.

## Build requirement

`scripts/build_llama_rocm_maxout.sh` now compiles for every distinct `gfx` target returned by `rocminfo`. On hybrid APU+dGPU systems this avoids accidentally building kernels for only one device.

## Topology capture

Run before performance testing:

```bash
export LLAMA_BUILD=$HOME/src/llama.cpp/build-rocm-maxout
bash scripts/collect_hybrid_memory_topology.sh results/topology
```

Preserve:

- host RAM/NUMA layout;
- ROCm agents and memory pools;
- AMD-SMI inventory/metrics;
- llama.cpp device list;
- PCIe/OCuLink negotiated speed and width;
- BAR/resource sizing;
- optional ROCm bandwidth-test output when installed.

## Memory-placement sweep

For Laguna:

```bash
MODEL="$LAGUNA_S21_Q4_MODEL" \
MODEL_ID=laguna-s-2.1-q4 \
EGPU_INDEX=0 IGPU_INDEX=1 \
CTX=32768 \
bash scripts/run_hybrid_memory_sweep.sh
```

Verify device indexes from `llama-bench --list-devices`; do not assume `0` and `1` on the physical host.

The sweep tests:

1. dGPU-only partial offload with multiple `-ngl` values;
2. iGPU/UMA-only when ROCm exposes it as a selectable device;
3. cross-device layer splits using multiple tensor ratios;
4. MoE-specific host placement using `--n-cpu-moe` and `--cpu-moe`;
5. direct-I/O model loading as a control for large hybrid allocations;
6. optional row/tensor splits when explicitly enabled.

## MoE placement hypothesis

For Laguna and Qwen3.5 MoE, generic CPU layer spill may be inferior to keeping expert weights host-resident while retaining attention/signal-path work on the discrete GPU. Current llama.cpp exposes:

```text
--cpu-moe
--n-cpu-moe N
```

The default sweep tries partial CPU-MoE placements and all-experts-on-CPU. Measure absolute TG/PP, TTFT, host CPU utilization, dGPU utilization, RSS, power and OCuLink/PCIe symptoms.

If host memory bandwidth is the bottleneck, `--cpu-moe` may fit the model but decode poorly. If activation transfers are small relative to expert-weight reads and useful compute remains on the dGPU, a partial setting may be a practical winner. Measure rather than assume.

## OCuLink-specific stability

Mixed AMD APU+dGPU ROCm systems have current reports of memory-access faults under some split modes. Therefore:

- layer split is the primary multi-device path;
- row/tensor split remains opt-in with `EXPERIMENTAL_SPLITS=1`;
- every crash/fault is preserved as evidence;
- repeat a promising split after process restart before promotion;
- never infer combined usable memory by simple addition.

## mmap vs direct I/O

Large UMA/hybrid model loads can behave differently under mmap/HIP allocation. The sweep includes a `--load-mode dio` control. Compare load success/time, resident memory, and runtime throughput rather than assuming direct I/O is universally faster.

## Context ladder for Laguna

Start at 8K/32K to identify viable placement. Only then test 64K/128K/256K. Long context consumes memory beyond the weight footprint; a model fitting at 8K does not prove it is viable at 128K.

For final candidates, preserve actual filled prompt tokens and derive decode throughput versus filled context.

## Promotion outputs

Return at least:

- fastest dGPU-partial placement;
- fastest UMA-only placement, if available;
- fastest stable APU+dGPU layer split;
- best `n-cpu-moe` placement;
- all-CPU-MoE result;
- direct-I/O vs mmap outcome;
- max stable context for each promising placement;
- p95/p99 latency and thermal/memory stability;
- exact command for the best Laguna operating profile.
