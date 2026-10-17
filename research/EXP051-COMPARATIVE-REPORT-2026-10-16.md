# EXP-051 — K2-Horizon-3.7B versus Qwen3.5-4B-Base

**2026-10-16 | 160 real BF16 CPU next-token model forwards, eight authored topics.**

Frozen protocol SHA256: `63c3c50afc2f98bb2d5928da180e7dc1f79f52b7672378ec829fb6f7209c4fdf`.
Model runner SHA256: `fdd90719644770e685aee84e82ea37b4add347723dce360458f6d668eb190977`.
Fixture receipt SHA256: `af188e9d490d420e02cc67d83a3405d37061fd281012f2dfb67efb8864740f9c`.

| Measurement | K2 3.7B | Qwen3.5 4B Base |
|---|---:|---:|
| supported original strict | 16/16 | 16/16 |
| supported swapped strict | 16/16 | 16/16 |
| supported withheld strict | 16/16 | 16/16 |
| unrelated original strict | 16/16 | 16/16 |
| unrelated swapped strict | 16/16 | 16/16 |
| Total strict | 80/80 | 80/80 |
| Positive evidence-switch topics | 8/8 | 8/8 |
| Positive question × evidence topics | 8/8 | 8/8 |
| Mean evidence-switch margin | 9.203125 | 9.355469 |
| Mean question × evidence interaction | 8.609375 | 8.703125 |
| Native CPU inference seconds | 48.172 | 53.494 |

## Frozen report digests

- `EXP051-K2-3P7B-RESULTS-2026-10-16.json`: `5e451b7da15c25e828dc667cad6cbc9e0ec9fde4524c9d3e44ff2b57f3aff687`.
- `EXP051-QWEN4B-RESULTS-2026-10-16.json`: `969c0258e18693709e6d49ae7ccbd2df38064562d9f4366c6b727516a9df073b`.

## Interpretation
- This is a **saturated synthetic QA pilot**: 8 distinct authored topics x 10 conditions each, not 80 independent facts.
- A/B label rotations and swapped/withheld evidence were held constant across models,
  but both succeeded; this task cannot rank semantic grounding or generalization.
- Mean margins are **within-tokenizer descriptors only**, not directly calibrated
  cross-family superiority tests. CPU timings are not controlled throughput benchmarks.
- Qwen is a pretrained Base checkpoint, K2 is separately developed, and vocabulary and
  architecture differ. No direct teacher lineage, hidden-state identity or neuron alignment.
- Different EXP-049/050 smaller-model QASPER results must not be conflated with these.
- Follow with human-adjudicated answer/foil pairs, unseen source-disjoint questions,
  free generation, citation attribution, hard negatives and paired bootstrap uncertainty.
- Draft research only. No weights changed, trained, merged, promoted, routed or deployed;
  no production NAS/Neo4j operations or arXiv submission.