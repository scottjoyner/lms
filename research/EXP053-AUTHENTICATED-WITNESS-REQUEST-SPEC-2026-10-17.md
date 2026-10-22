# EXP-053 authenticated witness request supplement — frozen 2026-10-17

This **supplements** (does not change) the frozen external high-water and SQLite
fixture protocols. Prepared before any approved scorer, owner enrollment, or
real EXP-053 forward. All real operator/witness trust roots remain unconfigured.

## Missing authorization boundary

The prior external-CAS request contains a nonce and SHA-256 identifiers but no
**verifiable operator authorization**. Anyone with network access to a future
backend could attempt to consume a sequence. A signed *response* is not request
authentication.

## Exact source-free request extension

Retain all v1 CAS request fields, with an additional REQUIRED field
`operator_authorization_b64`: canonical RFC 4648 Base64 of the *exact raw
bytes* of the private, independently signed operator authorization file. This
value must be at most 24 KiB Base64 and decode to 1..16,384 bytes.

The standalone witness MUST refuse a request unless all checks pass **before
BEGIN IMMEDIATE, any state read/transition, or any success signature**:

1. Exact request field allowlist, type-strict integer sequence >=1 and
   `expected_previous_sequence == sequence-1`, lowercase 64-hex nonce,
   predecessor SHA, claim SHA and scorer execution ID SHA.
2. Decode authorization as bounded, unambiguous UTF-8 JSON. Refuse duplicate
   keys, NaN, Infinity, unexpected/missing fields, non-canonical encoded
   public key or signature, and any extra potentially private passage field.
3. **Independently provisioned witness-side trusted operator public-key SHA**
   must be a pinned lowercase 64-hex digest, with no CLI selection. Verify the
   Ed25519 signature over canonical JSON excluding `signature_hex`.
4. SHA256 of the EXACT authorization bytes must equal request
   `claim_sha256`. Its `sequence`, `previous_claim_sha256`, and
   `runner_execution_id_sha256` must equal the outer CAS request exactly.
   This prevents a signed claim being substituted beneath unrelated counters.
5. Reject attempts to reuse an already committed nonce, claim SHA, execution
   ID or counter. Atomically commit high-water state before signing reply.

The client-side verifier must also refuse unless its separately
independently-approved expected witness-side operator key fingerprint agrees
with the actual approved operator root, and it must supply ONLY the exact
validated private authorization bytes. The scorer must not gain access to
the operator's private signing key. The Base64 payload carries signed
research metadata, NEVER RACE article text or private signing material.

The external witness response remains the previously frozen format and binds
the fresh client nonce, sequence, predecessor, claim SHA and execution ID.

## Authentication limitations

This is a **bearer proof of a pre-signed operator authorization**, not
end-to-end caller identity or a signature on the fresh request nonce. Anyone
able to steal a valid authorization may race its *one allowed* execution
claim. A real deployment STILL requires mutually authenticated transport,
short-lived/fenced credentials, independently owned operator custody,
a genuine off-host durable CAS, anti-rollback evidence, and an
operator-confirmed request lifecycle.

No production network adapter exists here. SQLite tests may use throwaway
signing keys and injected witness/operator pins ONLY inside disposable
test processes. All actual authority constants MUST remain EMPTY; the
default `_external_cas` MUST continue to deny. No model inference,
NAS/Neo4j writes, training or publishing new model measurements.
