# EXP-053 independently pinned operator request authentication — 2026-10-17

## Finding closed in research-only software

A future high-water CAS endpoint previously received SHA-256 values, a
fresh nonce and a claimed sequence but **no independently authenticated
operator authorization**. A signed witness RESPONSE does not authenticate
the party attempting to advance the witness's monotonic counter.

The prospective supplement was frozen BEFORE implementation:
`EXP053-AUTHENTICATED-WITNESS-REQUEST-SPEC-2026-10-17.md`,
SHA256 `8bc1c1c03ae5672c86e8df5dc1112e53a233902c32f2811fcaceb237f7f299a8`.
The already-frozen prior CAS wire protocol stays separately preserved.

Added `research/exp053_witness_request_auth.py`, an offline verifier
which requires a per-request proof of the **exact bytes** of the owner-signed
Ed25519 operator authorization: a bounded canonical Base64 field
`operator_authorization_b64`. It independently checks all CAS request
shape/type fields, SHA256 of those exact decoded authorization bytes,
a strictly pinned operator public key, the Ed25519 signature and the
authorization's sequence, previous claim and execution-ID binding.

This is deliberately stricter than checking an operator-provided hash:
a malicious caller cannot invent the authorization bytes or replay
a valid signature beneath a different claim/sequence/execution ID.

The real-source code `exp053_external_witness.py` now requires ANOTHER
approved trusted key fingerprint,
`APPROVED_WITNESS_OPERATOR_PUBLIC_KEY_SHA256=""`, and checks it matches
the previously approved operator signer. It validates the signed request
before attempting any external CAS. `exp053_operator_run.py` passes only
the exact private operator authorization bytes already inspected and
verified in the same file-descriptor read. No operator private signing key
is loaded into any scorer or witness client.

The isolated `tests/fixtures/exp053_sqlite_cas_fixture.py` now independently
verifies every received request with its own separately supplied test
operator key fingerprint BEFORE `BEGIN IMMEDIATE` and any SQLite mutation.
The fixture does not import any production signing secret. Its signing
key and operator key are **ephemeral synthetic test keys only**.

Adversarial tests refuse unsigned/missing proof, self-selected signer,
mismatched outer execution ID, claim SHA or sequence, invalid signature,
duplicated/malformed JSON, nonfinite/oversized payload, extra private
passage fields, boolean counters and bad Base64. The existing SQL
atomicity/concurrency/restart/rollback tests and end-to-end local
operator claim tests remain mandatory.

## Remaining scientific authorization gaps

The new supplement authenticates a pre-signed authorization as a
**bearer token**; it is not a live signature over the nonce or caller
identity. A thief who obtains a still-valid pre-signed authorization
could race its only allowed claim. A real external service MUST add
independently administered identity-aware transport (e.g. mTLS),
durable CAS, separate operator signing-key custody, request-lifecycle
fencing, off-host high-water, and native model execution evidence.

The actual production `_external_cas` STILL DENIES every request.
The witness-side operator pin, external witness signer/source pins,
human-review roster pin, weight scientific pin, scorer key/source pins,
and local operator signer/journal pins ALL remain EMPTY. Nothing was
deployed; no real independent witness, operator signing key, reviewer
approval, model forward, score report, NAS/Neo4j write or training.
PR #32 remains DRAFT and unmerged.
