# EXP-053 prospective attempt-ledger custody protocol — 2026-10-17

**Frozen before any native EXP-053 forward execution, human approval, or
approved scorer. This document must not be silently changed after freezing.**

## Purpose

A correctly signed attempt-log file digest currently binds arbitrary bytes,
not proof of the retry history or its relationship to model score rows.
For future scientific admission, the verifier shall additionally require a
*source-free per-cell attempt ledger* with a one-to-one pairing to each
score archive row.

## Exact private JSON shape

The top-level object has exactly these fields:

- `schema`: literal `exp053-attempt-ledger-v1`
- `execution_id`: the exact opaque ID in the independently signer-approved
  Ed25519 execution attestation.
- `packet_sha256`: the exact admitted candidate packet checksum.
- `entries`: an array of one record per **exactly one** score-row key
  `(model,case_id,arm,rotation)`.

Each entry has exactly `model`, `case_id`, `arm`, `rotation`,
`score_row_sha256` and `attempts`. The score digest is SHA256 of the
**canonical sorted JSON** representation of the source-free score row, using
the existing `exp053_race_review.canonical` function.

Each attempt has exactly:
- `sequence`: integer >=1 and contiguous starting from 1 within this cell;
- `status`: `failure` or `success`;
- `reason`: one of `none`, `timeout`, `runtime_error`,
  `tokenization_error`, `resource_exhausted`; success must use `none`,
  failure must use a non-`none` reason.

A cell must have 1–4 attempts with zero or more failures followed by exactly
one **final success**. Do not suppress failed attempts or treat an interrupted
cell as a successful measurement. All retries are subject to separately
reviewed research runner policy; the verifier does not authorize retries.

For each scored row, its cell key must appear in the ledger **exactly once**,
with exactly the same score-row SHA256. The ledger must contain no extra or
missing cells, even if valid scores could be synthesized to fill a gap.
The global expected count is `N approved articles × 3 models × 3 arms ×
2 rotations`, with N>=40. For the 64-candidate maximum, this is <=1152
cells. The JSON ledger is limited to 8 MiB.

Unknown fields, non-integer/boolean sequence or rotation, duplicate JSON
object keys, `NaN` / `Infinity`, duplicate cells, wrong ordering, missing
terminal success, extraneous successes, unknown failure reasons, mismatched
score hashes, wrong execution ID or packet ID, and altered private file bytes
must fail **HOLD**.

The ledger is a *private* local file, never a GitHub artifact. The existing
signed attestation's `attempt_log_sha256` covers its exact bytes. The verifier
must hash and parse that same checked descriptor, and recheck score+ledger
digests after computation before emitting a public source-free aggregate.

## Scientific limitations and authority

An agent can fabricate well-formed log entries and a matching score file.
Structural matching and Ed25519 signatures establish internal byte custody
and key provenance, **not** that a model executed, native logs are complete,
runtime inputs were actually those claimed or a novel run occurred.
A separately reviewed native scorer, independent physical model-load and
process witnesses, operator-held run-uniqueness/fencing evidence, and human
approval remain hard prerequisites for scientific promotion.

All four independent approval pins remain intentionally EMPTY.
Do not implement or run a neural forward script during this phase.
