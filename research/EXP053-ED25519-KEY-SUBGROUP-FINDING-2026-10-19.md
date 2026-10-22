# EXP-053 — weak reviewer public keys can defeat signature possession claims

**October 19, 2026 — technical security finding. NO_GO_RESEARCH_ONLY.**
All reproductions use invented, source-free review metadata and published
Ed25519 test vectors. No licensed article, actual signing key or real reviewer
was accessed. This is not proof that any real human review was compromised.

## Concrete synthetic reproduction

`exp053_reviewer_roster.load_roster()` previously accepted any distinct
base64-decoded 32-byte public-key strings. `cryptography`'s Ed25519 parser
also accepts compressed identity/low-order public keys. Using the Ed25519
identity-point key and a fixed publicly constructible signature, the local
OpenSSL-backed verifier accepted **the same signature for two different
messages**. A direct call to `exp053_race_review.verify_signed()` accepted a
fake reviewer record with this key. Since private-key possession is essential
for distinguishing independent reviewer identities, this is a serious
admission-integrity weakness, not merely input formatting.

## Source-only correction

The roster now checks that each public key decodes to a canonical curve point
in Ed25519's **prime-order subgroup**, is not the identity point, and has a
canonical compressed encoding. Checks include the Edwards25519 field bound,
square-root recoverability, canonical sign, and `[L]P == identity`, where
`L` is the prime subgroup order. This rejects small-order and mixed-torsion
keys, not only the all-zero and identity encodings. `verify_signed()`
independently enforces the same rule so direct trust-map callers cannot skip
it. No optional system crypto library is required at runtime; the source-only
validator handles public coordinates and not any secret scalar.

Regression tests include published RFC 8032 public points, fresh *ephemeral*
Ed25519 keypairs, noncanonical and weak points, a mixed-torsion point, a
synthetic keyless-signature demonstration, and direct-verifier refusal.

For independent numerical cross-check, 228 separate synthetic public-point
inputs (including 24 freshly generated Ed25519 keys and 200 random compressed
strings) were checked against the installed libsodium
`crypto_core_ed25519_is_valid_point` implementation. **All 228 classifications
matched**. This cross-check is evidence of local arithmetic consistency, not
a mathematical proof of implementation security or reviewer independence.

## Governance still blocked

The exact owner-approved reviewer roster pin is unset. There are zero
independently admitted human reviews, no 40 approved RACE articles, no
independent external witness authorization and no approved model inference.
PR #37 remains a **draft** for independent accountable technical review.
No credential custody, deployment, publication of private material or
scientific interpretation is approved by this patch.
