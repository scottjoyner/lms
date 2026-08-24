# Laguna-S-2.1 on RX 9070 XT 32 GB + RAM split — findings

Date: 2026-08-23 · Host: x1-370

## Setup

`poolside/laguna-s-2.1` — 256×4.5B MoE, 71.16 GB Q4_K_M (2-shard), loaded with
40% GPU offload: 66.28 GiB resident as ~30 GiB VRAM + ~36 GiB system RAM
(64 GiB was available). Context 32768, parallel 4.

## Throughput (plain)

| conc | aggregate tps |
|---:|---:|
| 1 | 3.1 |
| 4 | 16.3 |
| 8 | 17.4 |

RAM-side experts dominate: single-stream 3.1 tps is ~20× slower than
ornith-1.5 (66 tps) and unusable interactively. Aggregate 17.4 tps still beats
the CPU-only fleet nodes, but nowhere near daily-driver class.

## dflash speculation — does not work on stock runtimes

The `poolside-laguna-s-2.1-dflash` drafts (Q4_K_M 652 MB and MXFP4_MOE 1.19 GB,
copied from xwing) both fail identically under llama.cpp backend 2.29.1:

```
failed to load draft model '...dflash...gguf': error loading model:
done_getting_tensors: wrong number of tensors; expected 76, got 69
```

The dflash format appears to require poolside's own llama.cpp fork/runtime.
Both draft variants fail the same way; this is a format/runtime gap, not a
copy error. Follow-up if ever needed: install poolside's runtime extension, or
wait for upstream llama.cpp support for the dflash arch.

Also noted: `lms load --speculative-draft-*` CLI flags fall back to interactive
model selection even with `-y` (bug); speculation had to be configured via the
per-model default config (`user-concrete-model-default-config/poolside/
laguna-s-2.1.json`) where the failure at least surfaces a readable CAUSE.

## Conclusion

Laguna-S stays out of the daily-driver race on this hardware. Even with its
purpose-built dflash draft it would need poolside's runtime to be viable, and
the split-VRAM mode caps it around 17 tps aggregate. Ornith-1.5 remains the
production pick.
