# EXP-053 owner roster: require nonfollowing file access

**Research-only specification, frozen before code edits, October 19, 2026.**
Decision: NO_GO_RESEARCH_ONLY. A human study owner must independently enroll
reviewers and approve the exact fingerprint. The pin is still empty.

## Reproduced defect

`exp053_reviewer_roster.load_roster()` used
`getattr(os, "O_NOFOLLOW", 0)` and did not check that the flag is available.
A synthetic owner-pinned, unlicensed roster was accepted when the flag was
mocked as unavailable. The unprotected implementation could follow a swapped
final-component symlink between its `path.resolve()` preflight and `os.open`.
This is a fail-open kernel capability check, **not** evidence that any real
roster, signer identity or signing key has been compromised.

## Required narrow correction

- Refuse when `O_NOFOLLOW` or `O_NONBLOCK` is missing or zero. Never silently
  lower protections on systems without the required flags. Open the final
  roster component with both flags and keep a single fd for bounded read.
- From `fstat(fd)`, require owner uid matching the effective user, exactly
  one link, private regular-file type and no group/other mode bits. Refuse a
  hardlinked roster even if its bytes match the expected owner's hash. Do not
  change or chmod any file. The owner-custody reference and pseudonymous A/B
  human reviewer identities still require independent real-world validation.
- Preserve the exact SHA256 fingerprint, preregistered packet binding,
  double-checking of before/after dev/inode/size/mtime/ctime, duplicate JSON
  field rejection, slot uniqueness, Ed25519 public-key bytes, and rejection
  of roster files inside the public repository. Failure must remain sanitized
  `HOLD EXP053 ROSTER` and no protected source should be touched.
- Synthetic regression tests must simulate missing O_NOFOLLOW / O_NONBLOCK,
  symlink and hardlink targets, owner mismatch by mocking only `os.fstat`,
  and compare with one valid private synthetic roster. All public-safe test
  data use fake identities and synthetic 32-byte key placeholders, not actual
  reviewer keys, real forms, licensed RACE content, or model predictions.
- An exact valid-looking PIN or successful unit-test roster parse **does
  not** grant authority: maintain independent human enrollment, the
  non-reproducible source/witness gates, and the >=40 adjudicated article
  threshold. Do not write approval pins or enter model-inference execution.
