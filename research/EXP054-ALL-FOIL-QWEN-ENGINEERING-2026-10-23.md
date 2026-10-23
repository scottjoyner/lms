# EXP-054 — source-control and Qwen all-foil engineering checkpoint

October 23, 2026. Physical research host x1-370. Local-only development; no production changes.

## Frozen forward inputs, before scoring

The previously completed EXP-053 64 source-disjoint cases, full article identities and private exam-answer keys remain authoritative. The original machine-selected incorrect answer is no longer the only foil: a new separately frozen precommit contains all three incorrect exam-option labels for each of the same 64 sealed cases. This machine-selected foil set is NOT independently human-reviewed for semantic plausibility.

The new source arms are supported-prefix and unrelated next-article decoy-prefix, each scored under two fixed option rotations. A deterministic, tokenizer-only search considered native word-boundary source prefixes of 256–1600 Unicode codepoints, with a supported/decoy character retention ratio no less than 60%. Without consulting model scores or labels, the algorithm first minimized the largest supported/decoy token-count discrepancy across all three native tokenizers, then preserved maximum minimum source length and used deterministic tie breakers. The accepted maximum discrepancy was eight tokens across all tokenizer/rotation combinations. All 64 original articles passed.

Actual frozen tokenizer-only audit: 26/64 articles exact parity across all models/rotations; 56/64 maximum gap at most two tokens; 62/64 gap at most four; maximum gap in the entire cohort five tokens. These are NEAR-TOKEN-LENGTH controls, not fully length-identical data and not proof of semantic answer-evidence retention.

Private frozen control index (eight immutable parts) SHA-256:
d004bb8c1e7bbf1d0431da5a788defd9266a82f6e9ac0883491b342dd152a1ad

Private frozen three-wrong-option exam roster SHA-256:
63119e857f47d47c0ad21a20b50bcf9676448abf51e0f5741b2580b6b8abef2d

An initial equal-Unicode-codepoint-length control also passed source-custody checks, but yielded zero exactly matched input-token lengths out of 128 per model. It has an independent immutable private index SHA-256:
03c22cbacacc55f8407f46e753c7007934d0922930d8f52c1be1ece38a613ae4

## Actual new BF16 Qwen CPU forward pilot

Exact local Qwen3.5-4B-Base original BF16 checkpoint; CPU-only PyTorch 2.6.0+cpu and Transformers 5.20.1; HF network access disabled by offline flags; no remote custom model code. An independent resource-pressure check ran before loading the model. New private directory, 0600 case receipts, no overwrite or mutation of EXP-053 original results. For each article, 4 neural forwards cover 2 source arms times 2 answer-label rotations; all four answer letters are scored from the same logits, so the three predeclared wrong options require no extra forward.

The initial full eight frozen cases completed 32 neural forwards and had immutable private aggregate SHA-256:
8069c3e6b512002232f44abbe2933e04e66edd36d1246071f8496d7d20d431e4

The audited 16-case checkpoint completed 64 forwards and had immutable private aggregate SHA-256:
0fb129a9f776e66cfaa50ceb75954ea16016b9b7eefabec9007374c758ad1da0

Exploratory descriptive 16-case convenience-prefix results:
- Mean supported minus decoy gold-versus-foil margin averaged across all three wrong exam answers: +2.330078.
- 15/16 articles had a positive three-foil-averaged margin difference.
- 11/16 articles had positive source contrast for all three wrong answers.
- 11/16 articles had positive source contrast for all three wrong answers in BOTH answer-label rotations.
- 42/48 individual wrong-foil comparisons had a positive two-rotation averaged source contrast.

These are NOT representative, preregistered inferential or human-adjudicated semantic-grounding claims. Truncating an article might remove its required answer evidence; one must not infer that the gold-answer comparison remains semantically meaningful merely because its source prefix has matched length. The held-out article set is not proven absent from model pretraining. MiniCPM native visual understanding was not evaluated. No human code reviewer gate is required to keep source-only development iterating.

## Engineering safeguards

Independent source-only audit reconstructs every private source prompt from the originally frozen packet, validates source and decoy prefix SHA and exact input token counts, checks every label mapping and answer rotation against the precommitted roster, verifies runner/checkpoint/source hashes and finite answer log-probabilities, then writes an immutable private aggregate if all cases pass.

Relevant source code:
- research/exp054_prefix_length_control.py
- research/exp054_token_length_feasibility.py
- research/exp054_word_boundary_token_balance.py
- research/exp054_qwen_all_foils_cpu.py
- research/exp054_qwen_threefoil_audit.py
- tests/test_exp054_* for synthetic boundary, custody, tamper and rerun controls

All code committed in a CLEAN LOCAL research branch, no GitHub push or merge implied. The exact source/test suite at this milestone: 574 passed, 2 skipped, 276 passing subtests. All operator-owned private receipts remain local; licensed RACE text, answer keys, raw scores and weight shards are not included in this document or Git.
