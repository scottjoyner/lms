# EXP-053 — direct verifier trust registry canonicality

Frozen source-only specification, October 19, 2026. **NO_GO_RESEARCH_ONLY**.
No human review, independent key custody, model inference, or RACE passages.

## Reproduced synthetic defects

`exp053_race_review.verify_signed` currently calls `set(trust.values())`
before verifying each value's type and canonical encoding. Synthetic
reproduction with a disposable Ed25519 signer showed:

1. Two different base64 strings for the **same decoded Ed25519 public key**
   passed the distinct-key test and allowed an otherwise valid synthetic
   review to be accepted, because Python's base64 decoder tolerates unused
   nonzero padding bits in the last quantum.
2. A valid signer alongside an invalid/identity second public key was
   accepted; only the selected reviewer's public key was validated.
3. A list-valued second trust key raised raw `TypeError` instead of HOLD.

These are direct helper trust-map integrity defects, not evidence of a
compromised owner-approved roster. The separate `load_roster` validates
canonical base64 for all slots, but direct helper callers must also deny.

## Required behavior

- `verify_signed(review, trust)` must require 2 or 3 nonempty distinct string
  reviewer IDs and corresponding nonempty canonical base64 strings,
  each decoding to EXACTLY one 32-byte, canonical, nonidentity prime-order
  Ed25519 public point. Refuse a noncanonical textual alias and any duplicate
  decoded key material, not just duplicate encoded values.
- All malformed/unhashable trust values, incorrect encoding, weak second
  key and duplicated signers MUST yield sanitized `HOLD EXP053` *before*
  trying to import the optional signature provider, even on hosted CI.
- The cryptographic verifier must still validate a signature for the exact
  canonical unsigned review bytes and existing attestation/schema/role
  constraints; no signature can be assumed valid without a verifier backend.
- Require signed review's `signature` to be exactly 128 **lowercase hex**
  characters. `bytes.fromhex()` otherwise accepts whitespace and uppercase,
  permitting alternate textual receipts of the same 64-byte signature.
- Tests must use invented records and freshly generated disposable keys
  only; test the same-key alias, weak other key, unhashable value, duplicate
  key, valid distinct key pair, signature textual aliases, and absent optional
  crypto backend. Preserve green hosted CI. Keep PR #37 draft and owner/weight
  pins empty; do not merge, deploy or run models.

A valid pair of cryptographic keys still does not prove the owners are two
independent humans. 40 independently adjudicated articles and external
custody/authorization remain separate, unsatisfied gates.
