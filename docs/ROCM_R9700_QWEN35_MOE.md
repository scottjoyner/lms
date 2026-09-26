# Qwen3.5 MoE on the 32 GiB ROCm Target

Qwen3.5 MoE is a first-class comparison family in the R9700 maxout campaign.

## Primary target

`Qwen3.5-35B-A3B` is the most relevant single-GPU comparison target:

- 35B total parameters;
- approximately 3B active parameters per token;
- 256 routed experts;
- 8 routed experts activated per token plus one shared expert;
- 40 language-model layers;
- native 262,144-token context;
- trained with MTP.

The experiment must preserve the exact local artifact SHA and quantization. Do not assume an FP8, GGUF, FP4, or other conversion behaves like the upstream weights.

Prepared environment variables:

```text
QWEN35_35B_A3B_MODEL       primary Qwen3.5-35B-A3B artifact
QWEN35_35B_A3B_ALT_MODEL   alternate quant/conversion
QWEN35_122B_A10B_MODEL     optional 122B-A10B research candidate
```

## Why MoE needs separate analysis

A sparse model can have a large resident weight footprint while activating much less compute per generated token. Therefore report both total-parameter and active-parameter-normalized behavior where meaningful.

The key questions are:

1. Does the 35B-A3B model deliver materially higher TG than similarly sized dense/other-MoE artifacts once all weights are resident?
2. Does prompt processing scale differently from token generation because expert routing and hybrid attention stress different kernels?
3. Does concurrency improve GPU occupancy or create expert-routing/memory-bandwidth contention?
4. Does long context reduce the sparse-compute advantage because KV/context work dominates expert compute?
5. Does MTP produce a larger net gain on this model than draftless or external-draft speculation?
6. Which quantization yields the best quality-adjusted throughput within 32 GiB?

## Required comparisons

Run matched comparisons against Ornith candidates and dense Qwen candidates using the same:

- llama.cpp/ROCm build;
- context and filled-context depth;
- KV type;
- batch/ubatch;
- slot count;
- prompt/output lengths;
- sampling and chat template controls;
- power state.

Keep these metrics separate:

- PP tokens/s;
- single-stream TG tokens/s;
- aggregate concurrent output tokens/s;
- TTFT p50/p95/p99;
- VRAM and host RSS;
- watts and tokens/joule where available;
- quality score;
- MTP/speculative speedup;
- maximum stable context.

## MoE-specific derived metrics

For `Qwen3.5-35B-A3B`, calculate where useful:

```text
TG per active-billion parameters
PP per active-billion parameters
aggregate t/s per active-billion parameters
tokens/joule
quality-adjusted tokens/s
```

Do not use active-parameter-normalized metrics to hide the resident-memory cost. Always show absolute throughput and VRAM alongside them.

## Concurrency hypothesis

Because only a subset of experts execute for each token, `np=2` or `np=4` may improve hardware utilization if single-stream generation leaves compute resources idle. It may also hurt if multiple sequences increase expert working-set pressure or memory traffic. Measure rather than infer.

## Context hypothesis

At short context, sparse expert compute may dominate the performance advantage. At long filled context, attention/KV processing becomes a larger fraction of work. Plot TG against actual filled context for Qwen3.5-35B-A3B and compare the slope against Ornith and dense Qwen candidates.

## MTP

Upstream Qwen3.5-35B-A3B is trained with multi-token prediction. The local artifact/runtime must still prove compatible MTP support. Treat MTP as a separate loadout axis and sweep draft lengths 2/4/8/16 against the identical non-MTP baseline.

## Optional 122B-A10B candidate

`Qwen3.5-122B-A10B` is included as an optional research slot because a sufficiently aggressive local quant/conversion may be present. Do not expect it to fit a 32 GiB card by default. Startup failure/OOM is useful evidence; never spill or alter unrelated parameters solely to force a misleading comparison.
