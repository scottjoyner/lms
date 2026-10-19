# EXP-053 — refuse unapproved reviewer admission before private source access

**2026-10-19. Research-only, NO_GO_RESEARCH_ONLY.**

Reproduced with a disposable, PUBLIC-SAFE fixture: the `verified_admission`
entrypoint attempts to hash `SOURCE` before checking
`APPROVED_REVIEWER_ROSTER_SHA256` in `load_roster`. With the approval pin blank,
scientific admission cannot succeed, so touching a licensed source file, a
private packet, or a sealed answer key is unnecessary. This can waste I/O and
creates an unnecessary sensitive data access before human governance approval.
The reproduction did not open real licensed RACE text.

Minimum safe correction:

- At the FIRST line of `verified_admission`, reject any reviewer-roster pin
  that is not exactly 64 lowercase hexadecimal characters. Empty, malformed,
  wrong-type and stale-unset pins must return sanitized `HOLD EXP053` BEFORE
  stat/hash/open operations on any source/packet/answer key/reviewer form.
- This is *only* an early prerequisite check: a syntactically valid pin is
  not an approved human identity, roster, signature, or custody proof. Preserve
  all existing source and protocol digests, public manifest binding,
  downstream independently owner-pinned roster verification, signatures,
  A/B and third-party adjudication, and minimum 40 independently reviewed
  accepted cases. No bypass or fake admission path may be added.
- Tests must use only disposable public-safe synthetic files and mocks to
  prove the empty/malformed pin prevents ALL source/privacy reads. Add one
  control with a syntactically valid synthetic pin proving the real source
  authority check remains reachable and still returns HOLD for a mismatched
  source digest. No private RACE files, human keys or real reviews read.
- `gate()` must create no output on unapproved roster; no device access,
  model loading, promotion, Git publication or production deployment.

This protects a denial boundary; it does not itself authorize scientific
execution, independent human technical review, or custody of external state.
