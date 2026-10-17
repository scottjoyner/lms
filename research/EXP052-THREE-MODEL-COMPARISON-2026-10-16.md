# EXP-052 — MiniCPM-V 4.7-1B extension to EXP-051

**2026-10-16 | 240 real BF16 CPU text-only forward passes: 80/model.**

## Source custody

- EXP051 frozen audit and report digests remain unchanged.
- EXP052 frozen protocol SHA256: `d9d95a879f3c12a5168c86e7f24969e2e6ce92a6adcfc55b92b7cb4d6f651bee`.
- EXP052 executable SHA256: `e83093ab5c28527cbb5be6e5f71397898a60df8371155c0013d3f38032e7953c`.
- EXP052 result SHA256: `d9c69b06925e5f2b7beb5f93d6f1a49b12e84c219010d95aca9999f23fe2d475`.
- MiniCPM-V 4.7-1B HF revision: `286817f1132b998913a129a221c89b4402b4ab80` (full BF16 checkpoint pinned).

## Measured descriptive results

| Measurement | K2 3.7B | Qwen3.5 4B Base | MiniCPM-V 4.7 1B |
|---|---:|---:|---:|
| supported original | 16/16 | 16/16 | 16/16 |
| supported swapped | 16/16 | 16/16 | 16/16 |
| supported withheld | 16/16 | 16/16 | 16/16 |
| unrelated original | 16/16 | 16/16 | 16/16 |
| unrelated swapped | 16/16 | 16/16 | 16/16 |
| Total strict | 80/80 | 80/80 | 80/80 |
| Positive question × evidence | 8/8 | 8/8 | 8/8 |
| Mean within-model question × evidence effect (nats) | 8.609375 | 8.703125 | 15.750000 |
| CPU native forward time (seconds) | 48.172 | 53.494 | 12.381 |

## Required limits / next falsification

- All three models **saturate** this eight-topic authored synthetic test. No relative QA ability,
  human-independent semantic understanding, causal neuron or K2/Qwen teacher lineage follows.
- `NOT STATED` is a fixed provided option; no free response or citation attribution.
- MiniCPM is instruction-tuned/multimodal, yet this study uses raw text without its native
  chat template, no image frames. K2/Qwen Base have different size and training.
- Raw log-prob margins are not calibrated across tokenizers or runtimes. Seconds do not
  establish benchmark throughput; kernels, CPU load and model sizes differ.
- Stronger next cohort: independent human-adjudicated paper-disjoint supported and plausible
  unsupported answer alternatives, controlled foil length, swapped passage, wrong question,
  generation and abstention with paired article-level uncertainty.
- Draft-only: no weights, merging, live routing, production graph/NAS writes, or arXiv.