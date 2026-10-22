# Next-Generation Model Lanes

The benchmark should evolve by generation lane rather than by adding random models to one flat list. `benchmarks/nextgen_model_lanes.tsv` tracks candidates that are actionable, guarded, or research-only.

## Native 32 GiB lane

### Muse Glimmer 30B

Official GGUF artifacts exist for llama.cpp. Treat this as a high-priority native-R9700 control because it is explicitly designed for local autonomous-agent workloads. Record exact GGUF SHA, text/perception components, and any DFlash drafter separately. Require llama.cpp b10353 or newer for the official GGUF path.

### NVIDIA Nemotron 3.5 Lightning 30B-A3B

30B total / about 3B active, hybrid Mamba-2 + MoE + attention, up to 1M context, with MTP and released DFlash/DSpark speculative variants. It is an excellent architectural-diversity control against Qwen/Ornith and a native/small-MoE candidate. NVIDIA's optimized release is NVFP4; ROCm/llama.cpp compatibility must be proved rather than assumed.

## Frontier-MoE lane

### MiniMax M3

Roughly 427B total / 23B active, native multimodal, 1M context and MiniMax Sparse Attention. Because the full model is far outside local VRAM, enter through heavy quantization, pruning, CPU-MoE and hybrid-memory research rather than conventional full-resident inference.

### GLM 5.3

Guarded until public weights/license/runtime support are verifiable. The lane exists now so the scheduler can ingest the artifact immediately after release without redesign. Do not download unofficial lookalikes into the canonical NAS model library under this ID.

## Extreme compression lane

### Kimi K3

2.8T-class MoE with 1M context. This is not a normal benchmark candidate. The first jobs are metadata/tensor census, model-size arithmetic, quantization simulation, activation/router analysis where feasible, pruning/mixed-precision planning and predicted resident footprint. Only create a huge derived artifact when the admission planner predicts a plausible systems envelope.

## Reference controls

Keep a small/medium reference model in every campaign to detect host/runtime drift. GPT-OSS 20B/120B or equivalent verified artifacts can provide a same-lineage size control. Current Mistral/Gemma variants remain architecture-diversity controls only when exact local artifacts are identified.

## Generational comparison rule

For major families report three states when possible:

```text
previous generation
current generation
current generation optimized (quant/pruning/speculation)
```

Derive:

- task-success delta per GiB;
- quality delta at matched resident footprint;
- quality-adjusted TG/PP;
- quality-adjusted tokens/joule;
- max-context delta at matched host-memory tier;
- capability at the 32 GiB dGPU-only boundary;
- capability at the 64-96 GiB hybrid-memory boundary.

## Admission states

`ready-if-artifact-present`: verify SHA/provenance/runtime support, then add to executable model registry.

`ready-for-quant-research`: source weights exist but require a transform/placement study before benchmark admission.

`guarded-research`: metadata/planning only until size/runtime feasibility is established.

`guarded-until-public-weights`: no canonical artifact may be scheduled or downloaded until official public weights are verified.

## Execution status

Physical progress is independent from planning progress. Run:

```bash
bash scripts/benchmark_execution_checkpoint.sh .
```

The returned JSON status is one of:

```text
NOT_STARTED
PREFLIGHT_DONE
BUILD_DONE
BENCHMARK_STARTED
BENCHMARK_COMPLETE
```

Every agent handoff should include this JSON plus the complete path to any result tree. Pulling the branch alone is not execution evidence.
