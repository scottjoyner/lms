# HX-370 Heavy-Quant Large-Model Controls

This document adds heavily quantized GLM and DeepSeek Flash candidates to the HX-370 + 96 GB host-memory + OCuLink dGPU campaign.

## Why these controls matter

Laguna S 2.1 tests whether a roughly 70-75 GiB 118B/8B-active MoE can exploit host memory capacity plus discrete-GPU compute. Heavily quantized GLM and DeepSeek Flash models let us test the competing hypothesis: instead of optimizing placement of a ~70 GiB model, quantize a much larger model aggressively enough that its weights enter the same host-memory envelope.

The important variable is not only BPW. Measure how quantization changes:

- model weight footprint;
- remaining memory for KV/cache/buffers;
- fraction that can stay in dGPU VRAM;
- OCuLink traffic and partial-offload crossover;
- PP/TG throughput;
- p95/p99 latency;
- quality loss on the coding/agent suite;
- tokens/joule.

## GLM-4.5-Air heavy quantization ladder

Current llama.cpp GGUF community variants of GLM-4.5-Air include expert-focused quantizations in roughly the following host-memory range:

```text
~56.8 GiB  IQ3/IQ4 mixed expert recipe
~57.4 GiB  IQ3-class expert recipe
~60.0 GiB  IQ4-class expert recipe
~63.9 GiB  IQ4/Q5 mixed recipe
~67.8 GiB  Q4/Q5 mixed recipe
```

This family is therefore a strong control against Laguna because it leaves substantially more of the 96 GB host pool available for KV/cache and runtime buffers.

Prepared variables:

```text
GLM45_AIR_IQ3_MODEL
GLM45_AIR_IQ4_MODEL
```

Do not infer exact quant recipe from the registry label. Preserve the exact file name, SHA-256, average BPW if documented, and source repository.

## GLM-4.7-Flash

A generic heavy-quant slot is also included:

```text
GLM47_FLASH_MODEL
```

Use only a verified local artifact. Record exact quantization and model source. Treat it as a distinct family/release from GLM-4.5-Air.

## DeepSeek-V4-Flash

DeepSeek-V4-Flash is 284B total / 13B active with a 1M-token architecture context. The upstream instruct model already mixes FP4 expert weights with higher-precision non-expert weights, and llama.cpp-compatible GGUF conversions/quantizations exist.

The key aggressive quant candidates are:

```text
~82.0 GiB   IQ2_S
~95.4 GiB   IQ3_XXS-AS
~106.1 GiB  IQ3_XXS
~146 GiB    MXFP4/Q8 reference-style conversion
```

On a 96 GB host, IQ2_S is the principal viability candidate. IQ3_XXS-AS sits at the edge where OS/runtime/KV headroom may make it impractical without additional device placement. Reference MXFP4 is expected not to fit host RAM alone and is retained as a hybrid/offload control when available.

Prepared variables:

```text
DEEPSEEK_V4_FLASH_IQ2_MODEL
DEEPSEEK_V4_FLASH_IQ3AS_MODEL
DEEPSEEK_V4_FLASH_MXFP4_MODEL
```

## Quantization-quality experiment

For heavily quantized models, performance without quality is not useful. Run the same coding/agent suite as the main campaign and preserve raw outputs.

At minimum compare:

- patch correctness;
- tool-call/structured-output validity;
- instruction following;
- long-context retrieval;
- reasoning stability;
- repetition/degeneration failures;
- completion consistency across repeated runs.

The final analysis should calculate a quality-retention or task-success delta against a less aggressively quantized family baseline when available.

## Hybrid-memory sweep

Every large heavy-quant candidate that exceeds or approaches dGPU VRAM should also run through:

```bash
MODEL=/models/<artifact>.gguf \
MODEL_ID=<registry-id> \
EGPU_INDEX=<verified-index> \
IGPU_INDEX=<verified-index> \
CTX=32768 \
bash scripts/run_hybrid_memory_sweep.sh
```

For MoE models, retain the `--cpu-moe` / `--n-cpu-moe` sweep. Compare expert-host placement against ordinary partial layer offload.

## Memory-headroom gate

A model should not be called viable because its weight file is smaller than 96 GB. Promotion requires enough steady-state headroom for:

- OS and runtime memory;
- llama.cpp compute buffers;
- KV cache at the target context;
- tokenizer/server overhead;
- telemetry and concurrent request state;
- no persistent swap pressure.

Record `MemAvailable`, RSS, swap activity, OOM/reclaim behavior, and actual prompt context during tests.

## Primary cross-family questions

The final comparison should answer:

1. Does ~57-64 GiB GLM-4.5-Air outperform Laguna because it leaves more memory headroom and permits more aggressive dGPU offload?
2. Does ~82 GiB DeepSeek-V4-Flash IQ2 deliver enough quality to justify its much larger total model and tight memory envelope?
3. At the same host-memory footprint, is better intelligence obtained by a moderate quant of a ~110B MoE or an extreme quant of a 284B MoE?
4. Which model has the best quality-adjusted TG, quality-adjusted tokens/joule, and best 32K/64K agent throughput?
5. Where does OCuLink traffic dominate enough that a smaller model becomes faster despite lower nominal capability?

## Recommended initial order

Run the fastest-to-learn candidates first:

```text
GLM-4.5-Air ~57-60 GiB
Laguna S 2.1 ~70-75 GiB
DeepSeek-V4-Flash IQ2_S ~82 GiB
GLM-4.5-Air ~64-68 GiB
DeepSeek-V4-Flash IQ3_AS only if memory headroom permits
```

This sequence progressively tightens host-memory headroom and should reveal the placement/memory-bandwidth crossover before spending time on marginally viable configurations.
