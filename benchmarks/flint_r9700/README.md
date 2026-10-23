# Flint R9700 reproducible benchmark (LMS / K2-Horizon)

This benchmark captures **physical measurements**, not performance estimates, for [Flint](https://huggingface.co/unsloth/flint-GGUF) on Joyner's existing AMD Radeon AI PRO R9700 (gfx1201; 31.86 GiB reported VRAM).

**Current result, 2026-10-23:** the checkpoint and updated Vulkan runtime are valid, but neither quantization has completed a measured decode on this 16 GiB-RAM host with the existing inference services still running. The controlled IQ3 and IQ2 first-load attempts were stopped by host-memory safeguards. **Flint tok/s, TTFT, and coding-agent quality remain UNMEASURED**. The attached `results/2026-10-23_joyner_flint_preflight.json` is actual physical failure evidence; it must not be plotted as zero or compared with successful model decode rates.

## Immutable material

| Quant | Hugging Face bytes | SHA-256 | 2026-10-23 status |
| --- | ---: | --- | --- |
| `UD-IQ2_XXS` | 11,855,541,696 | `c1fa4a6d5ec67fbeff460c68829984e4aa3eea95dc9d2067691f19e5406deeee` | Fully downloaded, verified on x1 and Joyner; guarded loader abort, no token output |
| `UD-IQ3_XXS` | 14,535,427,520 | `7a0a095dde1d931063a5d571e19497d997edbc783eb1ed6b86529bfab9d94804` | Fully downloaded, verified on x1 and Joyner; guarded loader abort, no token output |

- Model repo: `unsloth/flint-GGUF`, revision **`20468edfc38ab43d6914cc88b08cbecb05d0fc62`**.
- Source runtime: `ggml-org/llama.cpp` commit **`970b51ce7c9ba5fff0726c5e9994f1c2f0e4eedc`**. Flint model architecture support verified in that checkout. Built with `GGML_VULKAN=ON`, portable x86 AVX2/FMA/F16C flags; actual runtime binary SHA-256 **`4cde6f61b370d344b0617e611c775bc4876d74d59116240fa00fc71edc6e15c5`**.
- Physical discrete card: `/sys/class/drm/card0/device/device = 0x7551`; Vulkan enumeration on Joyner: `Vulkan0` integrated Raphael, `Vulkan1` discrete Radeon gfx1201. Runner pins `Vulkan1`. GPU memory shown as 31.859 GiB, ~0.056 GiB used when idle.
- Original four `llama-server` processes remain untouched, Joyner localhost endpoints `:1234` and `:1236` returned HTTP 200 after testing. No BIOS change, swap reset, RAM purchase, inference promotion, production Auto-Router deployment or model replacement.

## Quickstart (authorized, existing hardware)

Copy a matching, already SHA-256-verified quant into the private Joyner experiment directory, with a separately compiled runtime and manifest. The code deliberately assumes this path rather than scanning/routing across a live fleet:

```bash
# On Joyner only: no model load, exit 2
python3 /home/scott/experiments/flint-r9700-20261022/run_flint.py --quant IQ2_XXS

# On Joyner only: one bounded test after a separate resources review;
# even a successful measurement exits 2 because it is not admission.
ALLOW_FLINT_R9700_BENCH=1 \
python3 /home/scott/experiments/flint-r9700-20261022/run_flint.py \
  --quant IQ2_XXS --profile smoke
```

Runner safeguards:

1. Exact host, GPU PCI ID and Vulkan device; exact model path, byte length, full SHA-256; immutable source revision and separately verified runtime binary; no symlinks or untrusted path substitutions.
2. Preflight requires ≥3.4 GiB **MemAvailable**, full-memory PSI avg10 below 1%, sufficient unused VRAM for full quant plus 2 GiB. After hashing, it retries a few bounded read-only checks (without relaxing thresholds). SHA scans issue `POSIX_FADV_DONTNEED` hints to limit pressure from the experiment's own page-cache reads.
3. `systemd-run --user --scope` unique experiment unit with `MemoryHigh=2500M`, `MemoryMax=3500M` and `TasksMax=96`. Monitor checks host available RAM, full-memory PSI and GPU VRAM each second. Experimental scope is stopped for available RAM below 2 GiB, full-memory PSI ≥5%, GPU free memory below 256 MiB, missing telemetry or elapsed profile maximum.
4. No default execution: `--run` and an explicit environment opt-in are required. Each attempt writes a create-only, private `0600` result containing real measurements or exact denial/abort reason. Only real `llama-bench` JSONL numeric timings are accepted as tok/s. No prompts, generated text or secrets are logged.
5. Profiles: `smoke` 64 prefill / 16 decode at most 100 seconds, `balanced` 128/512 prefill and 32/128 decode with two repeats, `throughput` 128/1024 prefill with 128 decode and three repeats. Performance profiles must not run before the smoke is green and resource headroom is established.

## Interpretation and future qualification

The 2.5B **active** parameters of this MoE do not represent full model residency: the checkpoint still occupies 11.86 GB or 14.54 GB. The existing 16GiB OS memory and near-full swap are concrete constraints even with 31.86GiB GPU VRAM. Both IQ2 and IQ3 tests passed preflight and whole-file checks, but their **runtime startup** crossed the host memory/pressure safety condition, so their process groups were stopped after roughly four seconds. No numeric Flint TPS was produced.

To compare Flint with Ornith, execute the **same prompts, tests, context size, sample count and harness** on an approved isolation window, log cold/warm TTFT, decode median/p95, correctness, structured tool-call success, memory and power; include all failures. Do not compare the prior eight-token Ornith short-sample ~9.51 tok/s from Destroyer to an unmeasured Flint run as a meaningful speed ratio.

### Known limitations

- The recorded Oct22/Oct23 abort receipts predate per-second peak VRAM telemetry; the updated runner now captures bounded per-second peak snapshots on future attempts.
- The live GGUF loader may require more than 3.5 GiB temporary host memory under unified page cache; this study **does not** assert that R9700 Vulkan cannot execute Flint with another memory configuration.
- Rootless transient cgroup memory control is an additional safeguard, not a guarantee of no host interference. Escalation to productive routing requires a separate acceptance process.
- Metrics depend on llama-bench's exact JSONL shape. Missing or malformed measurements are reported as **unmeasured**, never fabricated.

Tracking: [LMS #58 — Flint vs Ornith 1.5 vs Bonsai 2](https://github.com/scottjoyner/lms/issues/58). This is a research tool and **does not permit production deployment**.
