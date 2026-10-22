# EXP-053 — physical checkpoint revalidation (2026-10-22)

## Verified source and physical run

On `x1-370`, in a new detached isolated worktree at exact `scottjoyner/lms@dfbe6592bbdb6c5185ce81d62df3b01f6480b8db` (merged PRs #59 and #60 already included), the frozen `research/exp053_weight_identity.py` passed dry-run and full read-only safetensors byte hashing. The 43.68-second scan used low I/O/CPU scheduling, did not import/execute a neural model, and did not touch the existing dirty checkout.

The fresh JSON report was **exactly equal** to the frozen tracked `research/EXP053-WEIGHT-IDENTITY-2026-10-16.json` (all per-file SHA-256 values, sizes, and model receipt digests).

| Model | Shards | Bytes | Fresh full-file SHA-256 vs frozen manifest |
| --- | ---: | ---: | --- |
| K2-Horizon-3.7B | 36 | 10,116,554,080 | 36/36 match |
| Qwen3.5-4B-Base | 2 | 9,319,828,056 | 2/2 match |
| MiniCPM-V-4.7-1B | 1 | 3,109,054,480 | 1/1 match |
| Total | **39** | **22,545,436,616** | **39/39 match** |

Combined frozen and freshly verified checkpoint digest:
`62fb2c89fca1d64d6a154ecdf0712b97e79b8fc3c1eacacda1f3a98efb6d7462`

The locally retained fresh JSON receipt has SHA-256 `e95c771517869f2060b84c7a270bc3e59bd659500c5187d2227d2f94d00440b6`; its contents duplicate the existing tracked report. No private research dataset bytes are included here.

The isolated source-only test suite (`PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -B -m pytest -q -p no:cacheprovider tests/test_exp053*.py`) produced **407 passed, 2 skipped, 276 passing subtests** in 3.26 seconds.

## New actual loader compatibility finding

Offline, read-only `AutoConfig.from_pretrained(..., local_files_only=True, trust_remote_code=False)` in the existing isolated `ryzen-ai-venv` showed:

- **K2-Horizon-3.7B** requires custom local model code and cannot be natively loaded with `trust_remote_code=False`.
- **Qwen3.5-4B-Base** reports unsupported model type `qwen3_5`.
- **MiniCPM-V-4.7-1B** reports unsupported model type `minicpmv4_7`.

Runtime versions: `torch 2.6.0+cpu`, `transformers 4.57.6`, `accelerate 1.12.0`, `safetensors 0.8.0`, `tokenizers 0.23.0rc0`. No trust override, model load, download, or production change was attempted.

## Subsequent bounded Qwen synthetic neural-forward smoke

A separately qualified, **existing local** Transformers installation (`transformers 5.12.1`, `torch 2.13.0+rocm7.1`, `accelerate 1.14.0`) recognizes `Qwen3_5Config` and imports the native `Qwen3_5ForConditionalGeneration` class. This is a distinct local package path from the initial `ryzen-ai-venv` baseline above; neither environment was upgraded or modified for this check.

A CPU-only, offline smoke used `from_pretrained` on the frozen Qwen snapshot with `local_files_only=True`, `trust_remote_code=False`, `dtype=torch.bfloat16`, `device_map='cpu'`, `attn_implementation='eager'`, two CPU threads and a 160-second process timeout. Config/tokenizer hashes and shard sizes were rechecked against the frozen weight receipt before loading (the preceding **full byte-hash scan** had matched every shard). The synthetic, non-corpus prompt contained **13 tokens**. The model loaded, completed **one actual neural forward** (no generation), and returned **finite next-token logits with vocabulary dimension 248,320**. Model-load and forward elapsed approximately 1.7 s inside the process; outer shell process exited with code 0. Transformer optional fast kernels were unavailable; it reported the Torch fallback. This run used **no RACE article, private packet, labels, or model-scoring pipeline** and printed no answer text.

**Interpretation**: Qwen native BF16 synthetic CPU execution is now proven feasible on this physical host with the alternate installed runtime, but exact runtime package provenance still needs durable pinning, full per-prompt execution receipts and the preregistered machine-only dataset-arm runner. K2 and MiniCPM still lack a qualified native loader with this configuration. Do not confuse a single synthetic forward with RACE evaluation, cross-model comparability, or scientific grounding evidence.

## Next gate and limits

Finish **offline, isolated, pinned and audited** model-loader compatibility for K2 and MiniCPM and durable runtime pinning for Qwen on these **exact** checkpoints, then run a bounded, source-private machine-only pilot with complete failed-attempt receipts, resource limits and checkpoint/prompt provenance. Do not substitute different GGUF/quantized weights into the BF16 experiment.

**Scientific result: HOLD.** Exactly zero EXP-053 RACE/source-backed forwards or scientific model scores; one separate synthetic Qwen neural forward is documented above. Earlier tokenizer-only and machine-answer-key-proxy results remain distinct: machine-selected wrong-option foils are *not* human-adjudicated evidence. This is file-byte/source validation only; it does not prove model execution or semantic grounding. No licensed RACE passages/answers, private receipts, signer keys, or private model outputs are posted. No GitHub human approvals are needed for code integration, but this note does not close PR #32 or scientific issue #31.
