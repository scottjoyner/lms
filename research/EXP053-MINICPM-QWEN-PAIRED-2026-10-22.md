# EXP-053 — exact checkpoint machine-proxy comparison (2026-10-22)

**Status:** complete **machine-only exploratory** arms for Qwen3.5-4B-Base and MiniCPM-V-4.7-1B; **NOT** the original independently human-adjudicated research study. No causal, teacher-lineage, pretraining-disjointness or intrinsic model-superiority claims. [Scientific issue #31](https://github.com/scottjoyner/lms/issues/31), distinct draft PR #32.

## Physical reproducibility

- x1-370, local dedicated isolated research worktree; the Qwen scorer and tests were previously integrated in merged PR #62. These new MiniCPM sources build on that exact `main` snapshot.
- Private frozen **64 article-disjoint RACE question cases**, three passage controls (supported, withheld, next-article decoy), two fixed label rotations, a deterministic *wrong exam-answer option* machine foil. Do **not** call this foil independently semantically plausible or human adjudicated.
- Qwen3.5-4B-Base: **64/64 cases, 384 source-backed CPU forwards**, BF16 frozen checkpoint and local native Transformers 5.12.1. Case receipts and source are checked against original pinned manifest.
- MiniCPM-V-4.7-1B: **64/64 cases, 384 source-backed CPU forwards**, BF16 frozen checkpoint (3,109,054,480 bytes), native `MiniCPMV4_7ForConditionalGeneration` with Transformers 5.20.1 and Torch 2.6.0+cpu. No remote-model-code execution, model downloads or checkpoint substitution. Native raw-text path (no image inputs) so image-conditioned conclusions are unsupported.
- Per-model case receipts are owner-private, immutable, mode 0600; one receipt per case and 6 source/rotation rows each. **0 error receipts, 768 complete forwards across both models**. Final repeated source-alignment audit required both models to use **identical prompt SHA-256 and answer label identities** for every paired condition.
- Synthetic and source integrity tests: **459 passed, 2 skipped, 276 subtests passed** on an isolated checkout; includes 22 MiniCPM focused tests and 8 paired analysis synthetic tests. Private RACE source and keys were not uploaded into Git.

## Owner-private receipt identities

| Receipt | SHA-256 |
|---|---|
| Qwen full 64-case exploratory aggregate | `32a4a0168ffd38f6bf4790b6e39da4568f7f1d33a830ae1670843b1fdc84f0c8` |
| MiniCPM full 64-case exploratory aggregate | `bb87c24ae16d00d9e40b58d281ce92cc1aa21d3ae619402a3ec5756729f0fef8` |
| Same-source Qwen/MiniCPM paired comparison | `ad341c29a3ee06587278ed5d97e85f7d633683867e5718eb2dcf1f84a5d40c38` |

## Exploratory article-level descriptive score contrasts

All scores are log-probability margins for a frozen supported answer versus a **machine-selected** wrong exam answer, averaged over the two fixed option rotations. Values below represent the mean of the 64 article-level contrasts.

| Contrast | Qwen | MiniCPM | Paired Qwen − MiniCPM | Paired fixed-seed bootstrap 95% descriptive interval |
|---|---:|---:|---:|---|
| D: supported − withheld | +2.639648 | +1.329102 | +1.310547 | [+0.687500, +1.917480] |
| N: decoy − withheld | −0.281738 | −0.250488 | −0.031250 | [−0.500977, +0.461670] |
| S: supported − decoy | +2.921387 | +1.579590 | +1.341797 | [+0.699463, +1.978516] |

Bootstrap is a fixed-seed 2,000-replicate resampling of **paired articles**, not an independently preregistered human-study inference or adjustment for source/question selection. Distinct model tokenizers, text backbones and training distributions limit interpretation. Inference is CPU-only, offline, and source-private; no production services were modified.

## Remaining K2 safety and compatibility gate

K2-Horizon-3.7B custom sources are locally pinned (`configuration_k2_horizon.py` SHA-256 `5c2f993c1053d9462ebea6dea416c897fddfbb4a5edd904e486936b20d4badc5`, `modeling_k2_horizon.py` SHA-256 `fb09e010956bd51cfa7d4055b4381cff34c9e06164066b49e3546f38b2e6242f`). Restricted Docker container launched nonroot, read-only, network-disabled, without capabilities, and successfully imported CPU Torch 2.6.0+cpu and Transformers 4.57.6. K2 config did **not** load under 4.57.6: custom code expects the `PreTrainedConfig` API; a local native Transformers 5.20.1 installation has this API, but executing K2 custom code with that version has **not** been qualified, and a subsequent isolated invocation was blocked by safety checks. No K2 neural forwards or weights loaded; don't bypass the qualification gate.

## Code promotion policy

These new MiniCPM and comparison scripts are a **local-only review-free development candidate**, not deployed and not asserted to be merged in GitHub. GitHub branch creation was blocked, so do not claim public CI or merge completion. Tests and exact SHA-256 should be checked again at final branch head if GitHub publishing becomes authorized. Human *GitHub PR approvals* are not required; original human scientific adjudication requirements remain separate and unfulfilled.
