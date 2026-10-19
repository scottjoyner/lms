# EXP-053 — direct signed-review registry alias and malformed-key denial

**October 19, 2026. Source-only synthetic security finding.**
**NO_GO_RESEARCH_ONLY.** No actual human signing key, reviewer approval,
licensed RACE passage or inference was opened or used.

## Reproduced defects

`verify_signed(review, trust)` previously tested string-level uniqueness with
`len(set(trust.values()))`. The 32-byte Ed25519 key has one unused pair of
bits in the final base64 quantum. Python accepts a noncanonical alternate
base64 text with the same decoded key bytes when `validate=True` is used.
A synthetic experiment with a newly generated disposable Ed25519 key produced
**two different trust-map strings for the same decoded public key**, then
successfully verified a signed synthetic review as if it had a second signer.

The helper also verified a valid signer even when the *other* trust entry was
an identity/low-order key, and an unhashable list-valued trust entry caused
unsanitized `TypeError`. `bytes.fromhex()` also accepts upper-case and
embedded-whitespace aliases of otherwise identical signature bytes. These
are trust/receipt canonicality defects in direct helper invocations, not
proof of compromise of the separately owner-pinned private roster loader.

## Correction

The verifier now bounds the registry to the two independent slots and optional
third adjudicator, requires nonempty string reviewer IDs, canonical 32-byte
base64 Ed25519 keys, nonidentity prime-order subgroup membership **for every
trust entry**, and uniqueness of *decoded key material*. Any malformed,
duplicate or weak entry refuses with sanitized HOLD **before** the optional
cryptographic backend is imported. A signature receipt must have exactly 128
lowercase hex digits to prevent multiple textual receipts for identical
signature bytes. The existing unsigned canonical JSON, attestation, per-slot
role, Ed25519 signature, source packet and independent roster custody checks
remain mandatory. A valid signature does not, by itself, certify independent
human identity or blind review.

Source-only negative tests run even on hosted CI without the optional
cryptography dependency; a positive test uses disposable Ed25519 keypairs only
when available. No roster pin is set and zero scientific articles are admitted.
An independent reviewer must scrutinize this exact diff and run the tests.

Reproduce:
`python3 -m pytest -q tests/test_exp053_trust_registry_canonical.py`

**Still blocked:** real reviewer enrollment, 40 independently co-reviewed and
adjudicated source-distinct articles, independent observer/witness custody,
model-weight approval, production merge/deployment and model inference.
