# Activation-Guided Pruning Pipeline

This is a separate experiment axis from llama.cpp throughput tuning. The goal is to determine whether large dense/MoE checkpoints contain persistently low-value neurons, channels, layers, or experts that can be removed, zeroed, or more aggressively quantized without unacceptable coding/agent quality loss.

## Tooling note

A public repository literally named `Neuron Snooper` was not verified during experiment preparation. Do not block the work on that exact name. Equivalent/current tooling can be used for activation inspection and structured pruning, including activation viewers/debuggers, neuron-behavior graphing, and function-aware structured-pruning methods.

Any external tool added to the experiment must be pinned by repository URL + commit SHA and recorded in the evidence package.

## Core principle

Do not physically prune first. Use this sequence:

1. collect activation/router statistics;
2. identify candidate low-value structures;
3. perform reversible zero-ablation/masking;
4. rerun the exact quality + throughput benchmark;
5. prune/materialize only candidates that survive the gate;
6. convert/export to GGUF;
7. rerun the ROCm/OCuLink benchmark matrix.

This separates "this unit is rarely active" from "this unit can actually be removed safely."

## Calibration corpus

Activation collection must use representative workloads, not WikiText alone. Build a mixed corpus containing:

- coding generation and patch tasks;
- tool-call/JSON-schema tasks;
- repository-context prompts;
- long-context retrieval;
- repeated system-prefix agent workloads;
- reasoning/problem-solving samples;
- plain conversational/control prompts.

Store the calibration manifest and deterministic sample IDs with results.

## What to measure

### Dense/MLP models

Per layer/channel/neuron where practical:

- activation frequency above threshold;
- mean absolute activation;
- activation variance;
- top-k activation rate;
- contribution norm after output projection;
- gradient/sensitivity proxy when feasible;
- redundancy/correlation with nearby units.

### MoE models

Per layer/expert:

- router selection count;
- router probability mass;
- tokens routed per expert;
- expert utilization entropy;
- expert co-activation/substitution patterns;
- expert output norm;
- workload-specific specialization;
- cold-expert persistence across calibration subsets.

The most important MoE question is whether an expert is globally cold or merely specialized for a rare but high-value workload such as code/tool use.

## Candidate interventions

Test interventions separately:

1. expert zero-ablation;
2. expert pruning;
3. expert-specific lower-bit quantization;
4. MLP channel zero-ablation;
5. structured channel pruning;
6. redundant layer/block ablation only as a research arm;
7. attention-head pruning only when the architecture/tool supports it cleanly.

Prefer structure that existing kernels can exploit. Arbitrary unstructured sparsity may reduce parameter count without improving llama.cpp throughput.

## MoE pruning ladder

For large MoE candidates such as Laguna, Qwen, GLM, DeepSeek, Kimi, or MiniMax:

```text
0%     baseline
2.5%   conservative cold-expert mask
5%     light pruning
10%    moderate pruning
15%    aggressive research arm
20%+   only if quality retention remains unexpectedly strong
```

The percentage should be calculated per architecture-aware unit, not by blindly dropping the same expert IDs in every layer.

## Activation-guided mixed quantization

Pruning may not be the best first optimization. Use activation statistics to build a precision-allocation map:

```text
hot / quality-critical expert     higher precision
normal expert                     balanced quant
persistently cold expert          aggressive quant
critical attention/non-expert     preserve higher precision
```

This can reduce resident weight footprint without changing topology and may be more compatible with existing ROCm kernels than physical expert removal.

## Zero-ablation gate

Before producing a new checkpoint, run masked inference with a candidate manifest and require:

- no catastrophic generation degeneration;
- coding/task success within defined retention threshold;
- tool-call validity retained;
- no large long-context regression;
- no disproportionate failure on a calibration subgroup;
- measurable memory/compute rationale for materializing the change.

A unit that is rarely activated but causes a large quality drop when masked is not a pruning candidate.

## Quality retention gates

Suggested initial gates relative to the exact unmodified artifact:

```text
>= 99% task-success retention   conservative promotion
>= 97%                          experimental balanced candidate
>= 95%                          research-only unless performance gain is major
< 95%                           reject for production
```

Use task-level success rather than only perplexity.

## Materialization pipeline

For candidates that pass ablation:

1. generate a pruning/precision manifest;
2. apply changes to the source HF/safetensors checkpoint;
3. save as a new immutable artifact;
4. record source SHA + transform manifest + tool commit;
5. verify model load and architecture metadata;
6. convert to GGUF using pinned llama.cpp converter;
7. quantize according to the candidate precision recipe;
8. stage from NAS to SSD using the normal storage workflow;
9. run the complete benchmark/quality comparison.

Never overwrite the original checkpoint.

## Performance expectation

Physical pruning only matters if the runtime can exploit the smaller structure. Measure:

- file size / resident footprint;
- model load time;
- PP/TG;
- CPU-MoE behavior;
- dGPU-resident fraction;
- OCuLink traffic symptoms;
- maximum context;
- quality.

A 10% smaller model with identical decode speed may still be valuable if it enables much more KV/context or moves additional layers into dGPU VRAM.

## Required outputs

Each pruning experiment should return:

- source model identity/SHA;
- calibration corpus manifest;
- activation/router summary;
- candidate ranking;
- ablation manifest;
- pre/post quality results;
- transformed artifact SHA;
- post-transform GGUF quant identity;
- complete ROCm benchmark evidence;
- whether the gain came from throughput, VRAM/headroom, context, power, or all of the above.
