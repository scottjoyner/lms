# EXP-053 — independent reviewer public key group validation

Frozen source-only security specification, October 19, 2026.
**NO_GO_RESEARCH_ONLY.** This document must not be interpreted as proof of any
actual human review, signer control, owner enrollment, or scientific approval.

## Reproduction

The owner roster's old key validator enforced only canonical base64, 32
public-key bytes and uniqueness. The installed `cryptography` provider
accepts an Ed25519 identity-point public key. A synthetic identity-point
verification test accepts the same constructed signature for two different
messages; thus passing the verifier does not always prove private-key
possession. The private roster parser accepted this key with a matching
*synthetic* owner hash. The bug is fail-open weak-key enrollment, not evidence
that a real key, private RACE passage or reviewer has been compromised.

## Required fail-closed repair

1. Decode public keys as canonical 32-byte Ed25519 compressed points. Reject
   noncanonical y field encodings, points that do not exist on the curve,
   invalid x sign and the identity point.
2. Enforce membership in the prime-order Ed25519 subgroup: `[L]P == O`
   for Ed25519 subgroup order `L`. This rejects small-order and mixed-torsion
   points, not merely the identity's eight torsion relatives. Implement this
   using constant, source-only pure Python point arithmetic so the owner
   roster does not depend on an optional new native cryptographic library.
   This is validation of *public* input, not secret scalar operations.
3. Any malformed or weak point yields sanitized `HOLD EXP053 ROSTER` without
   reading or writing real reviewer keys. Existing 0600/no-follow fd guards,
   exact owner SHA and packet binding, A/B/ADJ slot uniqueness and downstream
   Ed25519 signature verification are not bypassed. The direct
   `exp053_race_review.verify_signed()` helper MUST independently enforce
   the same public-key subgroup rule, even if passed a manually supplied
   trust registry instead of an admitted owner roster.
4. Use RFC 8032 public test vectors and ephemeral synthetic keys for positive
   cases; reject identity, zero, noncanonical coordinates, sign ambiguity and
   synthetic mixed-torsion keys. Include a synthetic demonstration that the
   underlying provider accepting arbitrary-message identity signatures is
   blocked by the roster. No real signer identity, exam text or signatures.
5. Retest focused roster cases, EXP-053 suite and non-K2 suite. Preserve
   draft PR #37; do not merge, deploy, set enrollment pins or run inference.

A cryptographic group-membership check does not establish a real reviewer,
independent operator, blinded procedure, external witness or ability to admit
40 articles. **NO_GO_RESEARCH_ONLY** remains mandatory.
