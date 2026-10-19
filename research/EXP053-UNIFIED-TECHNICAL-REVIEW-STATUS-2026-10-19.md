# EXP-053 unified technical review / owner-readiness handoff

**Research-only. NO_GO_RESEARCH_ONLY. October 19, 2026.** This combined
change deliberately unifies two *source-only* paths under the exact frozen
remote GitHub PR #32 head:

`d14b1dcc1aa9d7626f8f6cd8d16e3ed8f330fa28`

1. The deterministic seven-document source-free custodian packet and its
   directory/path/partial-file/interrupt negative tests and review checklist.
2. The independent study-owner readiness CLI reporting publicly pinned cohort
   counts and **metadata only** for existing private local reviewer forms.

The first produces a **non-authorizing** ZIP; the second never opens RACE
licensed text, private reviewer forms, the answer key or any signing keys.
The fact that the original registered PR is draft, the owner approval
fingerprint is empty, and no independent human review has been admitted is
**not corrected** by unifying the code changes.

## Required independent technical review

- Compare the exactly named local candidate SHA and recorded SHA256 of the
  review patch against the precise PR #32 head, not main or a shallow clone's
  artificial history boundary. Review only the explicitly listed source-only
  changed paths and provenance/permissions prior to any public publication.
- Reproduce both synthetic suites in an isolated checkout:

  `python3 -m pytest -q tests/test_exp053_custodian_handoff_packet.py tests/test_exp053_review_owner_readiness.py`

- Inspect the source-only ZIP packager's required `NO_GO_RESEARCH_ONLY` and
  exit-code-2 behavior and all final-entry/no-follow/failure cleanup paths.
- Review the owner readiness metadata-only implementation, including refusal
  of unsafe parent-directory swaps and no-follow directory traversal, denial
  on missing source pins, and lack of any path that could grant human review
  approval or modify the private roster.
- Independently examine exact licensing, privacy and key-custody claims,
  then provide a reviewer-authored, attributable response out of band.
  Programmatically created reviews do not count as human independence.

## Still blocked

- Scientific review: no owner-approved A/B key-custody fingerprint, no 40
  independently adjudicated article-disjoint RACE cases; 64 source-free
  cohort records are candidates, not completed human reviews.
- Witness: no independently controlled signing/admin/restore root, no
  separately persisted high-water evidence, no native negative-control
  witness approval. Fleet SSH connectivity alone is not independent custody.
- Inference and release: NO model-forward passes, merges, GitHub publication,
  deployed service updates, key pin modifications or scientific approval.

**The source-only reviewer packet is NOT an approval object.** Only an
independent accountable human study owner/custodian can establish external
trust, consent and custody. Until then keep both the scientific workflow and
production deployment blocked.
