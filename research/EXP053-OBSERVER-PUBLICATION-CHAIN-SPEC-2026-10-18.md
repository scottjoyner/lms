# EXP-053 external-observer publication chain: offline negative verifier specification

Frozen **before implementing code**, October 18, 2026. This is a **read-only
research triage tool**, not actual independent custody, persistent high-water,
or scientific admission. Real publisher keys and trusted head pins are NOT
configured; no services, network endpoints or keys are to be created.

Inputs: (A) original frozen custody claims, (B) 1–12 signed receipt bundles,
(C) original frozen canonical receipt-reference JSON, (D) its exact SHA256
supplied as CLI literal, (E) one source-free observer publication JSON file
at most 32 KiB, (F) an Ed25519 observer public key in canonical base64,
(G) independent-looking expected public key SHA256 and (H) expected signed
publication head SHA256, supplied as *separate* CLI literals. Inputs F–H
are still **CALLER_SUPPLIED_UNTRUSTED** until independently pinned outside
the scorer. In particular, this verifier never writes approved signing-key
roots or enables a real external witness adapter. Files must be ordinary
non-symlink final components, read bounded from a single descriptor with
`O_NOFOLLOW`, `O_NONBLOCK`, and no source/private data printed.

All A–D must pass unchanged frozen `exp053_receipt_checkpoint_comparison.compare`.
The publication document must be canonical compact sorted JSON followed by
exactly one LF, UTF-8; `schema` exactly
`exp053-custody-observer-publication-chain-v1`, and exact fields
`schema`, `candidate_alias`, `plan_sha256`, `checkpoints`. It must bind to
plan/candidate from unchanged frozen planner. `checkpoints` is a list of 1–13
chronologically increasing entries, exact fields `sequence`,
`previous_statement_sha256`, `reference_sha256`, `observed_at_utc`,
`signature_hex`. Sequence MUST start with integer 0 and increase exactly one
at each entry, never boolean; max 12. The reference snapshot must have
`checkpoint_sequence` equal to the LAST signed entry sequence and its raw
SHA256 must match LAST `reference_sha256`. The last statement digest must
match separate CLI `--expected-head-sha256`.

Each `previous_statement_sha256` is 64 lowercase hex: all-zero digest in the
genesis statement, then SHA256 of canonical compact sorted JSON of the prior
complete **signed statement** (including its signature and bound context)
using `{"schema":"exp053-custody-observer-signed-checkpoint-v1",
"candidate_alias":...,"plan_sha256":...,"sequence":...,
"previous_statement_sha256":...,"reference_sha256":...,
"observed_at_utc":...,"signature_hex":...}`. The signer signs the same
structure *without* `signature_hex` using exact bytes:
`b"EXP053-OBSERVER-PUBLICATION-V1\\x00" + canonical(unsigned)`.
Observer key must be exactly 32 raw Ed25519 public bytes, canonical base64;
SHA256 of its raw public bytes must match the separate CLI
`--expected-observer-key-sha256`. The key MUST NOT be accepted merely from
a field within a publication file. Neither caller-supplied public key nor
digest proves a separate human owns or administers it.

All observation times are canonical `YYYY-MM-DDTHH:MM:SSZ`, valid UTC,
nondecreasing from entry to entry, and no more than 5 minutes ahead of the
local verifier UTC clock. No lower age bound is imposed on historical entries:
the local clock can be manipulated, so this is only a negative sanity check.
No automatic publication, network readback, key management, data repair or
side effects. Refuse malformed/oversized JSON, duplicate keys/fields,
unrecognized schema, invalid key/signature/chain, missing genesis,
nonmonotonic sequence/time, head mismatch, reference mismatch or absent pins.

Report exactly: `schema` (`exp053-observer-publication-chain-audit-v1`),
`candidate_alias`, `published_statement_count`, `observed_head_sequence`,
`head_reference_matches_local`, `observer_public_key_matches_caller_pin`,
`head_matches_caller_pin`, `publication_custody`,
`durable_replay_protection`, `scientific_execution_authorized`,
`independent_administration_approved`, `witness_durability_approved`,
`decision`. All matching flags can be true ONLY in the narrow local
consistency sense. Always return `publication_custody=CALLER_SUPPLIED_UNTRUSTED`,
`durable_replay_protection=NOT_PROVEN`, three approval flags false,
`decision=NO_GO_RESEARCH_ONLY` and CLI exit status **2** even with a valid
synthetic chain. Failed input prints only non-sensitive NO-GO status.

An attacker who can replace the observer key and both CLI digest pins can
rewrite and re-sign the whole chain; this tool MUST remain NO-GO even after
all local checks pass. Real rollback resistance requires independently
controlled protected keys, native durable CAS, a separately retained and
verified head, nonproduction rollback/recovery negative tests and human
custody review. Source-disjoint RACE label adjudication and exact model/scorer
bytes are separate remaining HOLD gates. Do not use this tool to deploy,
write NAS/Neo4j, contact fleet nodes, merge a draft or run model inference.
