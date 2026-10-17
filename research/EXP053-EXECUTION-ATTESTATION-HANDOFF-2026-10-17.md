# EXP-053 scorer provenance / signed execution receipts — 2026-10-17

## Newly closed software boundary

In addition to the frozen source, score, decoy, gold/foil and checkpoint checks,
scientific replay now calls `research/exp053_execution_attestation.py` before
parsing scores. An execution receipt must be privately stored, signed by the
**independently pre-approved scorer public key**, and bound to:

- exact human-approved candidate packet and signed admission;
- exact approved 39-shard weight inventory, all three model-specific checkpoint
  IDs, tokenizer-only preflight, and frozen prompt-builder SHA;
- exact private score-archive bytes and separate private attempt/retry-log bytes;
- exact expected score-row count (approved articles × three model families ×
  three source arms × two rotations), one raw/base-mode run and a source-code
  fingerprint for the independently approved scorer;
- a source-free execution-environment fingerprint and opaque execution ID.

The frozen prospective contract is
`EXP053-EXECUTION-ATTESTATION-SPEC-2026-10-17.md`, SHA256
`3564f784ec36f9e060b2b55b5375c07589595fa73f289a38caafa2865ca285b6`.

The verifier refuses caller-chosen keys, missing independent operator roots,
unapproved runner code, changed score/log files, simulated or chat-mode runs,
missing/private-leaking JSON properties, bad Ed25519 signatures, duplicated
JSON keys, symlinks/hardlinks, world-readable private files, and missing logs.
Input files are opened through single file descriptors with stable stat
checks, and individual-score bytes never go into public GitHub.
Replay additionally rehashes the score file after JSON parsing to reject
post-attestation mutations before statistical analysis.

The `exp053_verified_analysis.py` CLI now requires both
`--private-execution-attestation` and `--private-attempt-log` in addition
to its already-required private score archive. Only source-free hashes of
those attestations may appear in a future aggregate result.

## Authorization intentionally remains NO-GO

The **two additional operator pins** are intentionally empty:

- `APPROVED_RUNNER_PUBLIC_KEY_SHA256=""`
- `APPROVED_RUNNER_SOURCE_SHA256=""`

Likewise `WEIGHT_REPORT_SHA=""` and
`APPROVED_REVIEWER_ROSTER_SHA256=""` stay empty in their separate layers.
No runner implementation was built or approved; no true EXP053 neural forward
ever ran; no independent human reviewer approved any article.

The new Ed25519 tests create short-lived synthetic keys and dummy scores
*only to verify software refusal paths*; such signatures are not measurements,
independent human attestations, or evidence a model actually executed.

Even a correctly signed real receipt only proves possession of an approved
key and custody of claimed file bytes; it cannot prove honest log-probabilities,
actual model residency, complete retry history or physical-run provenance.
A separate accountable operator must inspect native model-loading and
runtime traces before any performance claims; corpus/answer copyright and
pretraining-contamination caveats remain unchanged.

No merge, model deployment, NAS/Neo4j writes, training or arXiv promotion.
