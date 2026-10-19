# EXP-053 offline reviewer validation parity — research-only specification

**Frozen before implementation, October 19, 2026; NO_GO_RESEARCH_ONLY.**

The existing private-only `exp053_offline_review.html` shows progress to two
real, independent blinded RACE reviewers. UI progress is helpful but **is not
scientific approval, signature verification, source-license authorization or
evidence of independent human participation**. No source passages or private
forms may appear in the public repository, test fixtures or tool output.

## Confirmed parity bug (synthetic only)

JavaScript `"ABCD".includes(value)` returns TRUE for `""`, `"AB"` and
other substrings; Python `value in "ABCD"` behaves the same way. The initial
independent test found **both** the UI progress counter and the trusted
`check_decision()` review-intake validator accepted empty and multi-character
labels. The Python acceptance could allow an invalid multi-character **foil**
to pass per-review validation and reach an otherwise valid primary agreement
(the primary supported answer is independently compared to the sealed key,
but the agreed foil was not checked for single-character identity there).
A synthetic Node reproduction yielded `complete=true` for empty and
multi-character supported answers. These cases are NOT reviewer evidence.

## Required fail-closed behavior

- For `accept` decisions, supported and foil choices must be **strings of
  exactly one character from {A,B,C,D}**, distinct. Empty or multi-character
  strings, numbers, `null`, arbitrary text or an answer equal to its foil must
  be incomplete, matching the authoritative Python admission predicate.
- The source quote must be a string with >=20 characters after trimming and
  must be an exact contiguous substring of the case's source article. The
  foil rationale must be a string with >=15 characters after trimming.
- For `reject`, `reject_reason` must be a string with >=12 nonspace
  characters, matching the Python gate. `unreviewed` always incomplete.
- The offline page must remain non-networked and use only local file handles;
  preserve browser CSP (`connect-src 'none'`), draft/export behavior,
  reviewer-blind answer display, and existing 64-case packet alignment.
- Repair the authoritative Python `exp053_race_review.check_decision()`
  boundary alongside the UI; its `accept` path must reject all blank and
  multi-character answer/foil strings before accepting any signed decision.
  Add independent Python negative tests and synthetic Node parity tests
  comparing JS `complete` with Python for positive and negative examples. Never load the actual RACE corpus or real review forms in tests.
- Apply the same exact single-character check to cohort answer-key
  selection (`exp053_race_review.cohort`) and downstream model-free
  receipt/metric validation (`exp053_verified_analysis.check_private_archive`
  and `exp053_metrics.analyze`). They must fail with a sanitized HOLD for
  malformed synthetic answer/foil labels, never implicitly accept string
  substrings or crash unexpectedly during label rotation. Add tests that
  exercise the validators with entirely synthetic sources and receipts.
- Tests may compare accepted/rejected validator behavior but **must not**
  synthesize human signers, owner roster pins or model-forward evidence.
  The public source HTML may be modified only after this protocol is recorded.

Completion is a **local visual indicator only**, never counted as an
independently signed review receipt. All preregistered scientific and
independent witness/custody gates remain NO-GO.
