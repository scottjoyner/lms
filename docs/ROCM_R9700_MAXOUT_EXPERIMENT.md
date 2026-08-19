# ROCm R9700 32 GiB llama.cpp Maxout Experiment

## Objective

Find the Pareto frontier for a single 32 GiB Radeon-class GPU running `llama.cpp` on ROCm/HIP: maximum stable context, highest prompt-processing throughput, highest generation throughput, lowest TTFT, and highest useful concurrent aggregate throughput without silently spilling critical work to CPU or accepting unstable/OOM configurations.

The experiment is evidence-first. Every run records the exact model artifact hash, llama.cpp commit, ROCm/GPU inventory, runtime flags, failures, server metrics, and raw benchmark output.

## Model status guardrail

- `Ornith-1.0-35B`: benchmark when a local GGUF is supplied through `ORNITH_MODEL`.
- `Qwen3.8-27B`: benchmark only when a real upstream-derived local artifact exists. As of experiment creation, public repositories with this exact name were placeholders and contained no model weights. Never benchmark a placeholder or infer results from an older Qwen release.

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
export ORNITH_MODEL=/models/Ornith-1.0-35B/<exact-quant>.gguf
# Only set this when genuine weights exist locally:
export QWEN38_MODEL=/models/Qwen3.8-27B/<exact-quant>.gguf
# Optional compatible draft model:
export DRAFT_MODEL=/models/<compatible-small-draft>.gguf
```

The runner hashes every supplied model with SHA-256. Quantization comparisons must be treated as different loadouts; do not combine their numbers under one model label.

## 3. Microbenchmark PP and TG first

The canonical `llama-bench` sweep measures prompt processing and generation independently. This is the fastest way to identify bad batch/ubatch/KV combinations before expensive server tests.

Baseline command:

```bash
~/src/llama.cpp/build-rocm-maxout/bin/llama-bench \
  -m "$ORNITH_MODEL" \
  -ngl all -fa on \
  -p 512,2048,8192,32768 \
  -n 128,512 \
  -b 512,1024,2048 \
  -ub 256,512,1024 \
  -ctk f16,q8_0,q4_0 \
  -ctv f16,q8_0,q4_0 \
  -r 5 --delay 1 -o jsonl
```

Important interpretation rules:

1. `llama-bench` PP/TG results intentionally exclude tokenization and sampling; they are kernel/runtime measurements, not end-to-end latency.
2. Keep PP t/s, TG t/s, TTFT/end-to-end server latency, and aggregate concurrent throughput as separate metrics.
3. Prefer the smallest KV type that preserves acceptable quality and yields material context/throughput benefit. Do not call q4 KV "free context" without quality testing.
4. A configuration that OOMs, falls back unexpectedly, or produces unstable repeated measurements is rejected even if one run is fast.

## 4. Establish the server baseline

Start with one slot, FlashAttention on, all model layers offloaded, q8 KV, and 32K context:

```bash
~/src/llama.cpp/build-rocm-maxout/bin/llama-server \
  -m "$ORNITH_MODEL" --alias ornith-1.0-35b \
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

The probe records TTFT, completion token count, per-request decode rate, aggregate output t/s, failures, and wall time.

## 5. Context frontier

Test contexts in increasing order:

```text
8K -> 16K -> 32K -> 64K -> 128K -> 256K
```

For each context, test `f16`, `q8_0`, then `q4_0` KV. Record whether the server starts, whether a full prompt+generation request succeeds three times, peak VRAM, system RAM, TTFT, PP performance, TG performance, and any fallback/offload messages. Stop declaring a context tier viable when it cannot complete the stability gate.

The published Ornith evaluations use long contexts (including 200K+ regimes), but that does not imply a 32 GiB local quantized loadout can practically sustain them. This experiment measures the local frontier instead of inheriting the model-card maximum.

## 6. Batch and ubatch tuning

Once a viable context/KV pair is found, sweep:

```text
batch:  512, 1024, 2048
ubatch: 256, 512, 1024
```

`ubatch <= batch` is mandatory. Compare PP t/s, TTFT, and peak VRAM. Larger batches are not automatically better; retain only configurations that improve the measured objective.

## 7. Speculative decoding

Always compare speculation against the same non-speculative baseline.

### Draftless n-gram

```bash
llama-server ... \
  --spec-type ngram-mod \
  --spec-ngram-mod-n-match 24 \
  --spec-ngram-mod-n-min 48 \
  --spec-ngram-mod-n-max 64
```

This is cheap to test and does not require loading a second model. It is workload-sensitive; code/repetitive text may benefit much more than unrelated prose.

### Small draft model

Only use a tokenizer/vocabulary-compatible draft artifact. Start conservatively:

```bash
llama-server ... \
  --spec-type draft-simple \
  --spec-draft-model "$DRAFT_MODEL" \
  --spec-draft-ngl all \
  --spec-draft-n-max 8 \
  --spec-draft-p-min 0.0
```

Sweep draft length `3, 5, 8, 12, 16` only after the baseline works. Record accepted draft tokens and end-to-end speedup. A faster draft model that consumes enough VRAM to force a worse target-model context can lose overall.

### MTP

Current llama.cpp exposes `draft-mtp`, but it is only valid if the selected model artifact/build actually contains and supports the needed MTP heads. Do not force this flag based on model-family naming. Enable the automated attempt with:

```bash
export ENABLE_MTP=1
```

A startup rejection is valid benchmark evidence, not a harness failure.

## 8. Concurrency

Test `-np 1`, `2`, then `4`, matching client concurrency to server slot count. Optimize two distinct objectives:

- interactive: lowest TTFT + strong single-request TG;
- throughput: highest aggregate output t/s under parallel load.

ROCm/HIP graph behavior with multiple slots has had recent reports of host-memory growth, so record process RSS/system RAM before and after long concurrency runs. Do not promote a parallel setting solely from a short burst benchmark.

## 9. One-command execution

Smoke/normal sweep:

```bash
export LLAMA_BUILD=$HOME/src/llama.cpp/build-rocm-maxout
export ORNITH_MODEL=/models/.../ornith.gguf
bash scripts/run_rocm_r9700_maxout.sh
```

Deep sweep including 256K and four slots:

```bash
DEEP=1 bash scripts/run_rocm_r9700_maxout.sh
```

Optional speculation:

```bash
DRAFT_MODEL=/models/.../draft.gguf ENABLE_MTP=1 DEEP=1 \
  bash scripts/run_rocm_r9700_maxout.sh
```

## 10. Required handoff evidence

Return the entire generated `results/rocm-r9700-maxout-*` directory, not screenshots or manually copied headline numbers. At minimum it must contain:

- runtime/GPU/ROCm snapshot;
- exact llama.cpp commit;
- SHA-256 and file metadata for each model;
- complete `llama-bench` JSONL;
- every server probe JSON and server log, including failed startups/OOMs;
- `amd-smi` snapshots and `/metrics` output where available;
- notes on thermals/power-limit changes or any external process using VRAM.

Do not tune clocks, voltage, power limits, ROCm environment overrides, or model quantization halfway through a comparison without starting a new loadout ID.

## Selection rule

Produce a Pareto table instead of one universal winner. At minimum report:

- fastest stable single-stream TG;
- fastest PP at 2K/8K/32K;
- lowest TTFT at 8K and 32K;
- largest stable context with >= 512 generated tokens;
- highest aggregate throughput at concurrency 2 and 4;
- best speculative speedup and acceptance behavior;
- best balanced profile that preserves meaningful VRAM headroom.

Only after those are measured should one profile be promoted as the default.
