# EXP-053 — three exact-weight model cohorts, exploratory machine-only completion

**Checkpoint:** October 23, 2026, approximately 03:41 America/New_York.
**Execution host:** x1-370.
**Status:** MACHINE-ONLY EXPLORATORY COMPLETE; ORIGINAL HUMAN-ADJUDICATED SCIENTIFIC EXPERIMENT **HOLD**.

## Reproducibility and source custody

Three separately pinned, **actual BF16 checkpoint** runs used the same frozen **64 unique, article-disjoint candidate RACE source cases**, three source arms per question (*supported*, *withheld*, *next-article decoy*) and two fixed option rotations. Every model scored 64 × 3 × 2 = **384 neural forward passes**; **1,152 actual forwards** in all, not tokenizer-only feasibility checks. Real model logits were used to score the original exam-answer key against a deterministic prechosen *incorrect exam option* as a machine-only proxy foil. A 64-article, 3-model readback required **identical prompt SHA-256, gold/foil labels, case identities, source-packet receipt hashes, model checkpoint identities and all six conditions** for each paired article.

The exact licensed article/question/answer payloads, per-case outputs, raw log probabilities, weights and owner-private admission packets remain **outside GitHub** on a secured local ext4 SSD under the existing 0700 EXP-053 private directory. Each model's 64 individual case receipts and aggregate report is immutable, 0600, source/checkpoint/runner-hash bound; there were **zero failure receipts**.

| Model | Model execution implementation | Articles | Forwards | Full-cohort receipt SHA-256 |
|---|---|---:|---:|---|
| Qwen3.5-4B-Base | Native BF16, CPU forced, Transformers 5.12.1 | 64/64 | 384 | `32a4a0168ffd38f6bf4790b6e39da4568f7f1d33a830ae1670843b1fdc84f0c8` |
| MiniCPM-V-4.7-1B | Native BF16 text-only path, CPU Torch 2.6.0 and Transformers 5.20.1, remote custom code disabled | 64/64 | 384 | `bb87c24ae16d00d9e40b58d281ce92cc1aa21d3ae619402a3ec5756729f0fef8` |
| K2-Horizon-3.7B | BF16 36-shard original, CPU Torch 2.6.0 and Transformers 5.20.1, pinned custom code in network-disabled, read-only, nonroot container | 64/64 | 384 | `0a5aaa8e18a8ec7f30206200aae9c979dbd14684d04d3d6cd72800e8ad2a1600` |

K2 loader identity included pinned custom code hashes and a separate **zero missing/unexpected/mismatched weight key** binding audit. K2's Hugging Face snapshot symlink resolved through an **exact separately mounted read-only CAS blob**; source-custody hashing was never disabled. Its restricted container used no network or Linux capabilities, read-only code/weights/packet/root mounts, a single dedicated writable private receipt subfolder, two CPUs and strict memory/PID/time bounds. Model metadata/docstring warnings for `k2_horizon` and `cache_position` were logged.

The K2 runner also enforced **a per-batch physical resource pressure gate**: 32 GiB minimum MemAvailable, low memory/CPU stall pressure, and bounded swap-in/out activity. Two gate holds occurred due to a short memory-pressure spike and transient available memory below 32 GiB; both refused to launch the next batch, and subsequent executions resumed only after a fresh passing physical observation. Immutable preexisting aggregates were **verified by canonical exact-byte comparison**, not overwritten. x1-370 showed no K2 model processes after the final audit.

## Descriptive complete 64-article machine-proxy results

Values are mean differences in *gold-answer versus deterministic wrong-answer log-probability margin*, after averaging the two fixed option rotations per article. **These are not calibrated across models to a common probability space or independently validated semantic foils.**

| Passage contrast | Qwen3.5-4B | K2-Horizon-3.7B | MiniCPM-V-4.7-1B |
|---|---:|---:|---:|
| D: supported minus withheld | +2.639648 (59/64 positive) | +2.452148 (60/64) | +1.329102 (45/64) |
| N: decoy minus withheld | −0.281738 (25/64) | −0.060059 (33/64) | −0.250488 (30/64) |
| S: supported minus decoy | +2.921387 (60/64) | +2.512207 (56/64) | +1.579590 (52/64) |

### Same-article paired model contrasts

Values are the difference of per-article passage-contrast values, and the descriptive 95% range comes from a **fixed-seed 2,000-resample article-paired bootstrap**. This was a secondary exploratory analysis, not the human-adjudicated preregistered inferential experiment.

| Pair | D paired difference, interval | S paired difference, interval |
|---|---|---|
| Qwen minus K2 | +0.187500 [−0.263184, +0.637695] | +0.409180 [+0.002930, +0.840820] |
| Qwen minus MiniCPM | +1.310547 [+0.663574, +1.928711] | +1.341797 [+0.708740, +1.968506] |
| MiniCPM minus K2 | −1.123047 [−1.734863, −0.497070] | −0.932617 [−1.556396, −0.254883] |

The N comparisons also show substantial uncertainty and are retained in the private canonical aggregate rather than cherry-picked from the report.

**Paired all-three-machine full-cohort report SHA-256:** `bd49d93c8847cc6e5c534b61b6a3dfd5fb11ef75f8e969be6f4004bf2ea601c8`.

### What this does not prove

- The wrong-option exam-key **proxy has not been independently human-adjudicated for semantic plausibility**; 64 machine-coded cases are **not** 64 reviewed human annotations.
- No human-blinded interrater agreement, approved foil adjudication, or original study's free-generation/abstention acceptance has occurred.
- No evidence here excludes pretraining contamination or identifies teacher-model ancestry, latent neuron identity, or causal grounding.
- MiniCPM ran a native **text-only**, not an image-grounded evaluation. Runtime/tokenizer/model-architecture differences constrain cross-checkpoint interpretations.
- These intervals are **descriptive exploratory bootstraps**, not corrected or preregistered confirmatory intervals. Do not present a ranking as general model capability superiority.
- No production serving/routing, fleet deployment, NAS/Neo4j writes, licensed-source redistribution, original draft PR #32 merge or journal/arXiv submission occurred.

## Code and validation disposition

- Qwen scorer and audit tests were merged in **PR #62** into GitHub `main`.
- The MiniCPM, K2 and three-model source-free research scripts and tests are retained on local `research/exp053-minicpm-native-local-20261022` branch. **This branch is not pushed or merged**; previous GitHub branch creation was blocked.
- The full locally isolated EXP-053 source/test suite: **507 passed, 2 skipped, 276 subtests passed**.
- K2 run launcher: `research/exp053_k2_isolated.sh`; K2 exact-source runner: `research/exp053_agent_k2_cpu.py`; full three-model audit: `research/exp053_agent_three_model_compare.py`; resource gate: `research/exp053_k2_resource_gate.py`.
- Scientific coordination and real physical checkpoint: [EXP-053 issue #31](https://github.com/scottjoyner/lms/issues/31). The independently human-adjudicated draft [PR #32](https://github.com/scottjoyner/lms/pull/32) remains a separate, still-blocked scientific track.
