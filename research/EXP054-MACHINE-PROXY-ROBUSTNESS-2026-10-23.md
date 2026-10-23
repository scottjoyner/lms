# EXP-054 — machine-proxy robustness audit, 2026-10-23

**Status:** Reused 3 × 64 original EXP-053 exact-weight source-backed case receipts; zero new model forwards. This is a secondary machine-only exploratory analysis, and does not reopen the closed [EXP-053 issue #31](https://github.com/scottjoyner/lms/issues/31) or superseded [PR #32](https://github.com/scottjoyner/lms/pull/32).

## Custody and implementation

Read-only verification of all 192 original immutable case receipts covered six source/rotation records each, model weight/tokenizer/runner identities, prompt SHA-256/answer-label parity across all models, and zero case failure records. New results were written exclusively into one new no-overwrite owner-private 0600 aggregate, not into the original receipts; no corpus text, keys, case-level values, or model weights were uploaded to Git.

- New analysis: research/exp054_machine_proxy_robustness.py
- New tests: tests/test_exp054_machine_proxy_robustness.py
- Full source-only suite: 521 passed, 2 skipped, 276 subtests passing
- Private report SHA-256: e8b5bb4f8d31d1b418bcbac4dd16cc4c0d4e49a42d4d99ff4a3c4c6fe56fbe4b

## Answer-option rotation instability is material

S is the machine wrong-exam-option gold-vs-foil log-probability margin under supported evidence minus unrelated source decoy evidence, averaged across two fixed label rotations. The *both positive* column requires both rotations separately to have S greater than zero. This is not a human adjudication or claim of semantic grounding.

| Model | Mean S | Article bootstrap descriptive 95% interval | Average S positive | Both rotations positive | Rotation sign differs |
|---|---:|---|---:|---:|---:|
| Qwen3.5-4B-Base | +2.921 | [+2.447, +3.379] | 60/64 | 56/64 | 5/64 |
| K2-Horizon-3.7B | +2.512 | [+2.030, +2.989] | 56/64 | 46/64 | 14/64 |
| MiniCPM-V-4.7-1B | +1.580 | [+1.043, +2.136] | 52/64 | 33/64 | 27/64 |

**MiniCPM is markedly more label-position-sensitive than its rotation-averaged result suggests.** For supported minus withheld D, rotation sign disagreements are 1/64 Qwen, 5/64 K2, and 30/64 MiniCPM. A positive average is not a stable invariant across both label rotations.

## Influence and prompt-length controls

| Model | S trimmed mean (remove 6 extreme articles per tail) | Leave-one-article-out mean S range | D trimmed mean |
|---|---:|---:|---:|
| Qwen | +2.962 | [+2.843, +3.033] | +2.779 |
| K2 | +2.563 | [+2.446, +2.590] | +2.376 |
| MiniCPM | +1.623 | [+1.512, +1.693] | +1.212 |

The S signal is not driven by one extreme article, but trimming or leave-one-out neither validates foil semantics nor excludes contamination.

**Critical length correction:** supported/withheld inputs differ by approximately 385.1 mean absolute tokens in Qwen/MiniCPM and 370.5 in K2. Supported/decoy inputs are **not token-length matched in any of 128 case×rotation comparisons** per model. Their average signed length difference is zero because positive and negative differences cancel. Absolute supported/decoy token gaps average **102.9** (Qwen/MiniCPM) and **98.5** (K2). Do not describe those controls as exactly length matched; they cannot rule out length or lexical relevance confounds.

Qwen and MiniCPM recorded identical input-token counts on all 384 prompts, despite distinct pinned tokenizer.json hashes and tokenizer implementation classes. A synthetic tokenizer test returned matching token IDs on one string. Compatible with shared tokenizer lineage, but not proof of general vocabulary or preprocessing identity.

## Recommended next agent-only experiment

1. Precommit alternative wrong-exam-option foils without selecting by model score; evaluate all three incorrect options as an explicitly new cohort with new no-overwrite receipts.
2. Freeze a genuinely case-by-case length-controlled source-decoy construction, with prompt and truncation hashes, before scoring.
3. Obtain documentary training-overlap evidence where possible and mark uncertain contamination as unknown, not excluded.
4. Evaluate native free generation and abstention separately rather than deriving semantic grounding from single-token likelihood.
5. Maintain scientific claim limits: no human-blinded review, no causal grounding, no teacher lineage, no unrestricted model superiority or production deployment claims.

The original human-adjudicated proposal is superseded and closed. No human GitHub PR code reviewers are needed for this agent-only research stream. No production changes were made.

## EXP-054 alternative-foil precommit, before new model inference

The source-only research follow-up is tracked at [EXP-054 issue #67](https://github.com/scottjoyner/lms/issues/67). A separate, owner-private machine-answer-key manifest has already been frozen for **all three alternative incorrect exam labels for every one of the 64 original article cases (192 alternatives)**. The selection is deterministic in fixed A/B/C/D order minus the sealed gold label, explicitly not selected or reranked by any model logit. Each original EXP-053 machine foil remains represented among the alternatives for comparability, with no re-adjudication claim.

- New preflight: research/exp054_all_wrong_option_preflight.py, with eight synthetic fail-closed tests.
- Owner-private immutable 0600 foil roster SHA-256: 63119e857f47d47c0ad21a20b50bcf9676448abf51e0f5741b2580b6b8abef2d.
- Full EXP-053/054 source suite: **529 passed, 2 skipped, 276 subtests passed**.
- New neural forwards: **zero**; no production routing, NAS/Neo4j, GPU/model changes, or licensed-source publication.
- Further scoring must use newly frozen, independently hashed experimental prompts and create new private no-overwrite model receipts. The future length-matched decoy control is not yet implemented or accepted. No inference should be initiated solely on the basis of this precommit.
