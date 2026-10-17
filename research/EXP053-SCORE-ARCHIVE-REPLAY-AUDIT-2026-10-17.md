# EXP-053 signed-only scientific replay: source and label custody hardening

**Date:** 2026-10-17. **Status:** research-only, unapproved, NO model forwards.

## Defect closed

The score-archive-specific preflight previously verified the prompt digest,
displayed option hash, tokenizer digest, and optional checkpoint digest, but
left direct checks of `article_sha256`, `context_article_sha256`,
`gold_label` and `foil_label` to the later statistical metrics module.
This had an unnecessary weaker trust boundary: callers to the lower-level
helper could pass a wrong source-article receipt or incorrectly shifted human
foil even when all prompt hashes were consistent.

The new `check_private_archive` performs these checks against each **exact
private source packet case** and the case's externally authenticated review
verdict, without seeing model weights:

- Source article SHA256 must equal the specific candidate's article bytes.
- Supported-context SHA256 must equal that article; withheld-context marker
  must be exactly `WITHHELD`; decoy-context SHA256 must equal the next
  **approved** case's passage by opaque ascending case ID, wrapping around.
- The original frozen question text, rotation-specific displayed option
  strings, prompt-builder output and option-order hash must agree exactly.
- Human gold and plausible foil labels must match the approved verdict.
  Rotation 1 maps each label by +2 places; neither label may be silently
  swapped with another plausible choice.
- All score records require the exact public-safe field inventory; any
  unrecognized field (including raw source passage text) is a HOLD.
- Every model requires an independently pinned checkpoint identity; callers
  cannot omit the weight map to bypass the check. Tokenizer identity, raw
  base-mode flag, and one-token answer choice lengths remain mandatory.
- Score replay reads the private packet from its **caller-supplied private
  directory**, rather than silently reading a fixed global packet from
  another location. Missing or malformed packet files raise an explicit HOLD.
  Its packet digest still must match the frozen public
  candidate manifest.
- The later `exp053_metrics.analyze` retains its independent complete-grid,
  cross-arm, duplicate-detection, logprob, rotation, sign-flip, bootstrap and
  Holm constraints.

## Software evidence and critical limitations

Added `tests/test_exp053_score_archive_binding.py`, using **40 entirely
fabricated, source-free cases** and no real human labels. Its 14 tests check
all six source/rotation combinations and deny article/context/decoy/option/
question/label/weight/tokenizer/token-count (including booleans) or schema substitution. Updated
the existing local-private test fixtures to include complete score receipts.
A single-record acceptance in the source checker only validates that record;
the pure later metrics analysis still enforces the complete 720+ row grid.

**No inference was run, and no fabricated score was approved.** A digest of
score rows is not proof that an inference engine actually computed their
log probabilities. A separate approved reproducible scorer and independently
attested execution receipt will be necessary before any scientific claim.

The actual verified, independently rehashed 39-file checkpoint archive remains
research evidence only. `WEIGHT_REPORT_SHA=""` and
`APPROVED_REVIEWER_ROSTER_SHA256=""` are still **EMPTY**. At least 40
independently admitted article decisions, real owner-reviewed signer enrollment
and deliberate checkpoint/scorer authority are unfulfilled. PR #32 remains
draft and unmerged; no NAS/Neo4j writes, model training or production changes.
