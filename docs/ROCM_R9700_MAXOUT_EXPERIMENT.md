# ROCm R9700 32 GiB llama.cpp Maxout Experiment

## Objective

Find the Pareto frontier for a single 32 GiB Radeon-class GPU running `llama.cpp` on ROCm/HIP: maximum stable context, highest prompt-processing throughput, highest generation throughput, lowest TTFT, highest useful concurrent aggregate throughput, speculative/MTP gains, and quality-adjusted model selection without silently spilling critical work to CPU or accepting unstable/OOM configurations.

The experiment is evidence-first. Every run records the exact model artifact hash, llama.cpp commit, ROCm/GPU inventory, runtime flags, failures, server metrics, and raw benchmark output.

## Model registry

The campaign is variant-driven through `benchmarks/rocm_r9700_models.tsv`. See `docs/ROCM_R9700_MODEL_VARIANTS.md` for the identity/provenance contract and instructions for adding additional quants or Ornith-1.5 variants.

Prepared slots are:

```text
ORNITH10_35B_MODEL       Ornith 1.0 35B baseline
ORNITH15_PRIMARY_MODEL   verified Ornith 1.5 primary candidate
ORNITH15_ALT_MODEL       verified Ornith 1.5 alternate candidate
QWEN38_MODEL             verified Qwen3.8-27B candidate
```

The original `ORNITH_MODEL` variable remains accepted as a compatibility alias for `ORNITH10_35B_MODEL`, but new runs should use registry-specific variables. Optional artifacts are skipped when absent. Never benchmark a placeholder or silently substitute another generation.

Every result belongs to the identity tuple:

`family -> generation -> variant -> artifact SHA-256 -> quantization -> MTP capability -> llama.cpp SHA -> ROCm build -> runtime configuration`

## 1. Build llama.cpp for the actual GPU target

Do not hard-code a gfx target from the product name. The build script obtains it from `rocminfo`, then compiles the current checkout specifically for that target.

```bash
bash scripts/build_llama_rocm_maxout.sh
```

Equivalent core build command:

```bash
GPU_TARGET="$(rocminfo | awk '/^[[:space:]]*Name:[[:space:]]*gfx[0-9]+/{print $2; exit}')"
HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
cmake -S ~/src/llama.cpp -B ~/src/llama.cpp/build-rocm-maxout \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_HIP=ON \
  -DGGML_HIP_GRAPHS=ON \
  -DGGML_HIP_ROCWMMA_FATTN=ON \
  -DGGML_CUDA_FA=ON \
  -DGPU_TARGETS="$GPU_TARGET" \
  -DGGML_BACKEND_DL=ON \
  -DGGML_CPU_ALL_VARIANTS=ON \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=ON \
  -DLLAMA_BUILD_SERVER=ON \
  -DLLAMA_BUILD_TOOLS=ON
cmake --build ~/src/llama.cpp/build-rocm-maxout -j"$(nproc)"
```

Record `llama-server --version`, `llama-bench --list-devices`, `rocminfo`, `amd-smi static`, and the llama.cpp commit before interpreting any result.

## 2. Supply exact local artifacts

```bash
export ORNITH10_35B_MODEL=/models/Ornith-1.0-35B/<exact-quant>.gguf
# Set only for verified exact artifacts:
# export ORNITH15_PRIMARY_MODEL=/models/Ornith-1.5/<exact-artifact>.gguf
# export ORNITH15_ALT_MODEL=/models/Ornith-1.5/<alternate-artifact>.gguf
# export QWEN38_MODEL=/models/Qwen3.8-27B/<exact-quant>.gguf
# Optional compatible draft model:
# export DRAFT_MODEL=/models/<compatible-small-draft>.gguf
```

The runner hashes every supplied model with SHA-256 and copies the registry into the evidence directory. Add additional variants as unique registry rows; do not combine different quantizations or conversions under one model label.

## 3. Microbenchmark PP and TG first

The canonical `llama-bench` sweep measures prompt processing and generation independently:

```bash
~/src/llama.cpp/build-rocm-maxout/bin/llama-bench \
  -m "$ORNITH10_35B_MODEL" \
  -ngl all -fa on \
  -p 512,2048,8192,32768 \
  -n 128,512 \
  -b 512,1024,2048 \
  -ub 256,512,1024 \
  -ctk f16,q8_0,q4_0 \
  -ctv f16,q8_0,q4_0 \
  -r 5 --delay 1 -o jsonl
```

Keep PP t/s, TG t/s, TTFT/end-to-end latency and aggregate concurrent throughput as separate metrics. Reject OOM, unexpected fallback and unstable configurations even if one repetition is fast.

## 4. Establish the server baseline

```bash
~/src/llama.cpp/build-rocm-maxout/bin/llama-server \
  -m "$ORNITH10_35B_MODEL" --alias ornith-1.0-35b \
  --host 127.0.0.1 --port 8080 \
  -ngl all -fa on \
  -c 32768 -np 1 \
  -b 2048 -ub 512 \
  -ctk q8_0 -ctv q8_0 \
  --cont-batching --metrics --perf --jinja
```

Then probe it:

```bash
python3 scripts/llama_server_probe.py \
  --endpoint http://127.0.0.1:8080/v1 \
  --model ornith-1.0-35b \
  --concurrency 1 --requests 5 \
  --prompt-repetitions 256 --max-tokens 512 \
  --label baseline-32k-q8 \
  --output baseline-32k-q8.json
```

## 5. Context and KV frontier

Test increasing context tiers:

```text
8K -> 32K -> 64K -> 128K -> 256K
```

At each tier test `f16`, `q8_0`, then `q4_0` KV. Record startup success, full prompt+generation completion, peak VRAM, system RAM, TTFT, PP/TG and fallback/offload messages. KV quantization is not assumed quality-neutral.

## 6. Batch and ubatch tuning

Sweep:

```text
batch:  512, 1024, 2048
ubatch: 256, 512, 1024
```

`ubatch <= batch` is mandatory. Compare PP t/s, TTFT and peak VRAM.

## 7. Ornith 1.0 -> 1.5 matched comparison

Before enabling speculation or changing quants, compare every verified Ornith-1.5 candidate against Ornith-1.0 using the same:

- llama.cpp commit/ROCm build;
- context and filled-context workload;
- KV type;
- batch/ubatch;
- slot/client concurrency;
- sampling and prompt/output shape;
- GPU power/clock state.

This matched baseline is the only clean estimate of the generation/artifact delta. Quantization, MTP and speculation are separate axes.

## 8. Quality gate

Run the same coding/agent suite for Ornith 1.0 and each 1.5 candidate. Preserve raw outputs and scores for instruction following, patch correctness, structured/tool-call reliability, repeated-prefix workloads, long-context retrieval/use and completion stability.

A faster model can win the throughput profile while failing the balanced/quality profile.

## 9. Speculative decoding and MTP

Always compare speculation against the same non-speculative loadout.

Draftless n-gram baseline:

```bash
llama-server ... \
  --spec-type ngram-mod \
  --spec-ngram-mod-n-match 24 \
  --spec-ngram-mod-n-min 48 \
  --spec-ngram-mod-n-max 64
```

Compatible draft model:

```bash
llama-server ... \
  --spec-type draft-simple \
  --spec-draft-model "$DRAFT_MODEL" \
  --spec-draft-ngl all \
  --spec-draft-n-max 8 \
  --spec-draft-p-min 0.0
```

MTP is opt-in:

```bash
export ENABLE_MTP=1
```

Only registry rows whose `mtp` field is not `false` are attempted. A startup rejection is valid evidence. For Ornith 1.5 report native generation delta, MTP incremental delta, speculation incremental delta and each mode's VRAM/context cost separately.

## 10. Concurrency

Test `-np 1`, `2`, then `4`, matching client concurrency to slot count. Optimize interactive latency and aggregate throughput as distinct objectives. Record process RSS/system RAM alongside VRAM during longer parallel runs.

## 11. One-command execution

```bash
export LLAMA_BUILD=$HOME/src/llama.cpp/build-rocm-maxout
export ORNITH10_35B_MODEL=/models/.../ornith-1.0.gguf
# export ORNITH15_PRIMARY_MODEL=/models/.../ornith-1.5.gguf
# export ORNITH15_ALT_MODEL=/models/.../ornith-1.5-alt.gguf
bash scripts/run_rocm_r9700_maxout.sh
```

Deep sweep:

```bash
DEEP=1 bash scripts/run_rocm_r9700_maxout.sh
```

Optional speculation/MTP:

```bash
DRAFT_MODEL=/models/.../draft.gguf ENABLE_MTP=1 DEEP=1 \
  bash scripts/run_rocm_r9700_maxout.sh
```

## 12. Required handoff evidence

Return the entire generated `results/rocm-r9700-maxout-*` directory. It must preserve runtime/GPU/ROCm snapshots, exact llama.cpp commit, the model registry, identity metadata/SHA-256 for every artifact, complete `llama-bench` JSONL, all server probe JSON/logs including failures, `amd-smi` snapshots and `/metrics` output where available.

Do not change clocks, voltage, power limits, ROCm overrides or model quantization halfway through a comparison without starting a new loadout identity.

## Selection rule

Produce a Pareto table rather than one universal winner. Report at minimum fastest stable single-stream TG, PP at 2K/8K/32K, lowest TTFT, max stable context, concurrency 2/4 throughput, speculative/MTP benefit, Ornith 1.0->1.5 matched deltas, quality deltas and the best quality-adjusted balanced profile.
