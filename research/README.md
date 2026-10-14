# K2-Horizon model-lineage layer matrices

Research-only model comparison, conforming to LMS separation of
**exact model identity, observational similarity, and behavioral quality**.
This is not an inference endpoint, a new benchmark run, or a deployment gate.

`k2_layer_overlap.py` reads SHA-pinned prior experiments EXP-016, 018, 019,
021 and produces a full human-readable Markdown layer matrix and behavioral
timeline in the **central knowledge repository**, not stand-alone results JSON.
It validates every 5×5 pool/role CKA slice (all 25 cells), preserves
undefined early token states as undefined, and computes:

- Same-depth average and final-depth CKA for every adjacent K2 stage.
- Row nearest-neighbor mapping, mutual nearest sampled depths.
- Maximum mean bijective sampled-layer matching on *measurable* cells.
- Separate masked-mean and last-token, whole-response and signed-role grids.
- Cross-architecture K2↔Qwen3.5-0.8B CKA matrices.
- Exact K2 checkpoint identifiers, actor/recipient behavior and
  answer-format/abstention evidence, without conflating metrics.

## Repeatable read-only analysis

```bash
R=/home/scott/knowledge/50-Research/llm-research/latent-world-geometry-2026
python3 research/k2_layer_overlap.py --evidence-dir "$R" \
  --out "$R/EXPERIMENT-022-LAYER-OVERLAP.md" --replace
python3 -m unittest -v tests.test_k2_layer_overlap
sha256sum "$R/EXPERIMENT-022-LAYER-OVERLAP.md"
```

Expected report SHA256:
`4cbdbdf96a597097b2ef8be66d6b82ec502a16ad316abdc10ac19a32c48bbaf0`.

This code uses **previously executed** frozen model runs;
it does not load a model, generate embeddings or reexecute experiments.
The `~/knowledge` directory contains original source scripts,
fixtures, numbered historical protocols, reports, and secondary
machine evidence. The original HF weights must be independently
restaged at the pinned revisions for fresh neural measurements.
No auto-download, NAS operation, production service effect,
fine-tuning, graph mutation or model-promotion authority.
