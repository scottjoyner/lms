# ROCm R9700 Model Variant Registry

The ROCm maxout campaign is model-registry driven. The canonical registry is:

`benchmarks/rocm_r9700_models.tsv`

Each row represents one independently identifiable benchmark artifact. The runner reads the environment variable named by `model_env`, hashes the file, copies the registry into the evidence directory, and runs the same PP/TG/context/concurrency/speculation sweep.

## Identity rule

Treat this tuple as the benchmark identity:

`family -> generation -> variant -> artifact SHA-256 -> quantization -> MTP capability -> llama.cpp SHA -> ROCm build -> runtime configuration`

Never aggregate results across different artifact hashes or quants under a single model result.

## Prepared model slots

```text
ORNITH10_35B_MODEL       Ornith 1.0 35B baseline
ORNITH15_PRIMARY_MODEL   verified Ornith 1.5 primary candidate
ORNITH15_ALT_MODEL       verified Ornith 1.5 alternate candidate
QWEN38_MODEL             verified Qwen3.8-27B candidate
```

Unset optional variables are deferred, not failures.

## Add another Ornith 1.5 artifact

If the host contains another 1.5 quant, conversion, checkpoint, or architecture variant, add a unique row instead of overwriting an existing slot. Example:

```tsv
ornith-1.5-35b-q4km	ornith	1.5	35b-q4km	Q4_K_M	ORNITH15_35B_Q4KM_MODEL	auto	false	<exact-source-or-provenance>
```

Then export the path:

```bash
export ORNITH15_35B_Q4KM_MODEL=/models/ornith-1.5-35b-q4_k_m.gguf
```

Use a neutral variant ID when provenance is uncertain. Do not infer an official name from a filename.

## Matched generation comparison

The clean Ornith 1.0 -> 1.5 delta requires identical runtime conditions:

- same llama.cpp commit;
- same ROCm build and GPU power state;
- same context and filled-context workload;
- same KV type;
- same batch/ubatch;
- same slot count/client concurrency;
- same sampling and prompt/output shape;
- speculation disabled.

After that baseline, measure MTP, n-gram speculation, draft-model speculation and alternate KV/quant settings as separate axes.

## Quality comparison

Performance promotion must be paired with the same coding/agent suite for every candidate. Keep raw outputs and scores for instruction following, patch correctness, structured/tool calls, repeated-prefix workloads, long-context retrieval and completion stability.

A faster Ornith 1.5 artifact may win the throughput profile while losing the balanced/quality profile. Preserve both conclusions.

## Example execution

```bash
export LLAMA_BUILD=$HOME/src/llama.cpp/build-rocm-maxout
export ORNITH10_35B_MODEL=/models/ornith-1.0-35b.gguf
export ORNITH15_PRIMARY_MODEL=/models/ornith-1.5-primary.gguf
export ORNITH15_ALT_MODEL=/models/ornith-1.5-alt.gguf

bash scripts/run_rocm_r9700_maxout.sh
DEEP=1 bash scripts/run_rocm_r9700_maxout.sh
```

The complete `results/rocm-r9700-maxout-*` directory is the evidence artifact. Do not reduce the return package to headline token rates.
