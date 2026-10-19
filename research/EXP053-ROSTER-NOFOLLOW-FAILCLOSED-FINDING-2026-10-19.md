# EXP-053 — owner roster must refuse unsupported no-follow file operations

**October 19, 2026. Source-only negative-test finding. NO_GO_RESEARCH_ONLY.**

## Reproduction

`exp053_reviewer_roster.load_roster` previously constructed an open flag with
`getattr(os, "O_NOFOLLOW", 0)`, silently dropping no-follow protection when
an OS/Python environment does not provide it. A disposable synthetic roster,
with invented A/B pseudonyms and artificial public-key bytes, was successfully
parsed when the `O_NOFOLLOW` flag was mocked as unavailable. As a consequence,
a final-component symlink swap between `path.resolve(strict=True)` and
`os.open` could be followed under that environment. This does not demonstrate
an actual compromised independent human or a cryptographic break.

## Narrow correction

The loader now explicitly rejects the absence of `O_NOFOLLOW` or `O_NONBLOCK`
with a sanitized `HOLD EXP053 ROSTER` rather than degrading to path-following
reads. After opening the single nonfollowing fd it also validates that the
regular file is owned by the effective user and has exactly one link, in
addition to the existing mode-0600, bounded-size, stable device/inode/content
and exact pinned SHA checks. The pinned packet identity, source-root exclusion,
blind-slot independence, key uniqueness and downstream Ed25519 verification
are preserved.

A new synthetic test module checks the normal private single-link success
case; deny-only missing OS flags before opening the file; hardlinks, symlinks,
wrong effective-user ownership, malformed owner pins and public-repository
placement. All data are disposable fake JSON; no licensed RACE article, human
private key, real owner roster or output approval was accessed.

## Limits and necessary independent work

These local filesystem checks do not attest human identity, external witness
custody, no adversarial same-user tampering, or the trustworthiness of every
ancestor directory of the roster path. Tests do not authorize changing the
owner pin or signing on behalf of a reviewer. The study still requires two
independently enrolled and blinded humans, 40 genuine source-distinct
co-reviewed cases, adjudication of disputes, signed receipt approval,
independent witness/rollback evidence and separately approved model weights.
This source-only fix is NOT a production or science release approval.

Reproduce: `python3 -m pytest -q tests/test_exp053_roster_nofollow_failclosed.py`
