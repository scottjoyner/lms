# EXP-053 prospective custody-receipt consistency audit — research only

**Frozen before implementing the parser.** This source-free, read-only audit
checks that *untrusted* synthetic/future claimed challenge receipts bind to an
unchanged twelve-case plan. This is NOT an independent witness, a source of
trusted time, authority admission, signed human approval, or durable replay
protection. It cannot promote any scientific or operational gate.

Input A: exact source-free custody claims manifest from
`exp053_custody_boundary_audit.parse` (<=16 KiB); generator from
`exp053_custody_negative_challenges.plan`. The expected plan digest is the
SHA256 of canonical compact sorted JSON UTF-8 serialization of that generated
plan, `json.dumps(sort_keys=True,separators=(",",":"))` (ASCII default).

Input B: source-free receipt bundle JSON (<=48 KiB), canonical duplicate-key
rejection and strict type/field checks. Exact top-level keys:
`schema`, `plan_sha256`, `candidate_alias`, `observer_public_key_b64`,
`checkpoint_reference_sha256`, `receipts`. Schema is
`exp053-custody-negative-receipt-bundle-v1`. `plan_sha256` must exactly
match input A's generated plan digest; `candidate_alias` must exactly match
its alias. `checkpoint_reference_sha256` is nullable only with zero receipts,
otherwise it must be lowercase 64-hex digest. The checkpoint reference is
self-supplied and UNTRUSTED: it is never queried or checked for durability.
Observer public key is similarly null only with zero receipts, otherwise
canonical base64 of exactly 32 raw Ed25519 public-key bytes; the key is
UNTRUSTED and MUST NOT be installed into approved witness/operator/runner
pins. Empty receipts are valid, meaning no evidence, not verification.

Each of at most 12 receipts has EXACT keys:
`scenario_id`, `outcome`, `nonce_hex`, `evidence_sha256`,
`observed_at_utc`, `signature_hex`. Scenario must exist in generated plan and
occur at most once. Outcomes: `DENIED`, `BYPASS`, `UNKNOWN`, all untrusted
assertions; `BYPASS` must be surfaced as a risk claim. `nonce_hex` is 64
lowercase hex and unique across rows. `evidence_sha256` is a 64-character
lowercase SHA256 (no private evidence bytes in input). `observed_at_utc` is
strict `YYYY-MM-DDTHH:MM:SSZ`, normal Gregorian time, with no leap second.
`signature_hex` is 128 lowercase hex characters encoding a 64-byte Ed25519
signature over:

`b"EXP053-CUSTODY-NEGATIVE-RECEIPT-V1\\x00" + canonical_json(unsigned)`

Where `unsigned` has EXACT keys `schema` (value
`exp053-custody-negative-receipt-v1`), `plan_sha256`, `candidate_alias`,
`checkpoint_reference_sha256`, `scenario_id`, `outcome`, `nonce_hex`,
`evidence_sha256`, `observed_at_utc`. The bundle's only claimed observer public
key must cryptographically verify every row; failure is a hard HOLD. A valid
signature means only self-supplied-key authenticity, not independent custody.

Perform bounded local clock sanity: receipt time must be no more than 24h
before the verifier's local UTC clock and no more than 300s in the future.
The local clock is not an independent trusted clock. This is only a stale/
future *screen*; it cannot prove fresh acceptance, external checkpoint
publication, high-water persistence, or defeat operator-side rollback.

No raw path, source text, private evidence SHA, observer key/fingerprint,
nonce, signature or timestamp is printed. Report only stable schema,
`NO_GO_RESEARCH_ONLY`, `scientific_execution_authorized=false`,
`independent_administration_approved=false`, `witness_durability_approved=false`,
`signer_authority=UNTRUSTED_SELF_SUPPLIED_KEY`, number signed, number missing,
scenario IDs missing, claimed DENIED/BYPASS/UNKNOWN counts, and
`all_challenges_independently_attested=false`. Even 12/12 fresh signatures
and zero claimed bypasses are NON-AUTHORITATIVE.

Refuse unreadable/nonregular/symlinked/oversized/duplicate-key documents,
malformed or additional fields, wrong plan/candidate, duplicate scenarios or
nonces, invalid Ed25519 signatures, stale/future date, and missing crypto
library. Every CLI invocation returns code 2, including a parsed report.
No remote access, key generation, installation, model inference, NAS/Neo4j
writes, Git merge, approval pins, external high-water state or witness
creation are authorized. Tests may use self-generated synthetic software keys
only in disposable in-memory fixtures; those keys are not operators or trusted
independent custodians.
