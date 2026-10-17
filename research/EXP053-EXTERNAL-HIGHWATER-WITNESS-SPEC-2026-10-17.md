# EXP-053 external high-water CAS witness — prospective research contract

Frozen October 17, 2026, **before implementation and without any live witness or model inference**.

## Required independent trust

The existing local operator journal and used-ID markers detect duplication only
while their filesystem has not been rolled back. An approved operator must
provision an independently administered, durable atomic compare-and-swap (CAS)
witness. The witness's private Ed25519 signing key and committed state **must
never reside in the local scorer, owner journal directory, or research repo**.

The future source-reviewed research gate requires:
- `APPROVED_EXTERNAL_WITNESS_PUBLIC_KEY_SHA256`: SHA256 fingerprint of a
  verified raw Ed25519 public key, independently pinned in reviewed code;
- `APPROVED_EXTERNAL_WITNESS_ADAPTER_SHA256`: separately reviewed pinned
  implementation digest for a real authenticated external CAS adapter.
Both are intentionally EMPTY until actual separate operator review. The
live adapter does not exist in this phase. Tests MAY inject a simulated CAS
backend; no public CLI/user-selected endpoint can select a trusted witness.

Witness public key must be distinct from the operator and scorer keys.

## One signed response per fresh CAS request

After the local operator authorization and locked journal sequencing have
been verified, but **BEFORE** any local used-ID marker or journal mutation,
the verifier generates 32 cryptographically random bytes for a unique nonce.

The external adapter must atomically compare its own previously committed
high-water sequence and prior claim SHA256 to the request, then persist the
next sequence and SHA256 of the whole signed operator authorization, and
return a signed response. Stale/forked/wrong-sequence requests must be
refused and **must not** be attested as committed.

Request: exact JSON shape:
- `schema` = `exp053-external-highwater-cas-request-v1`
- `nonce_hex` = 64 lowercase hex characters generated anew per attempt
- `sequence` = strictly positive integer
- `expected_previous_sequence` = sequence - 1
- `previous_claim_sha256` = SHA256 of previous authorized claim
- `claim_sha256` = SHA256 of current whole operator-signed authorization
- `runner_execution_id_sha256` = SHA256 of current scorer execution ID

Signed response: exact fields:
- `schema` = `exp053-external-highwater-cas-response-v1`
- `nonce_hex`, `sequence`, `previous_claim_sha256`, `claim_sha256`,
  `runner_execution_id_sha256` copied from this unique request
- `witness_counter` = committed monotonic sequence, matching `sequence`
- `status` = exact string `committed`
- `witness_public_key_b64` = canonical raw 32-byte Ed25519 key
- `signature_hex` = Ed25519 signature over canonical sorted JSON excluding
  the signature field.

Refuse malformed/unknown/missing fields; untrusted witness key; signature
failure; wrong nonce; wrong claim, sequence, counter or execution ID; wrong
status; noninteger/boolean counters; and any unavailable/out-of-date witness.

A signed response with matching fresh nonce is evidence that a distinct
keyholder asserted a CAS commitment, **not** proof that a remote system
actually durably stored it. Native independent audit and durable backend
implementation are required. The trust model explicitly forbids a
dummy in-memory test witness from acting as a real authority.

## Failure ordering and durability

The remote CAS is intentionally committed first. Only after signed response
verification may the local claim marker be created and journal fsynced.

If external CAS succeeds and the process crashes before local update, the
external high-water is *ahead*, and all subsequent attempts fail closed
until a separate authorized recovery audit. Never reset external high-water
or delete used-ID markers automatically to force a retry.

If local operator journal and markers are both rolled back, a *genuinely
independent* durable high-water witness rejects the reused sequence; a
falsified or rolled-back simulated witness cannot provide this property.

Only public-safe witness receipt SHA256/counter are allowed in future
aggregates. The signed response and native evidence remain PRIVATE.

## Scientific HOLD

No real external witness/adapter or operator authority is configured.
There are no accepted independently human-reviewed articles, no approved
runner/score receipts, and zero EXP053 forward passes. This phase must not
start model inference, change runtime routing, deploy services, write
Neo4j/NAS, or claim replication. PR stays draft/unmerged.
