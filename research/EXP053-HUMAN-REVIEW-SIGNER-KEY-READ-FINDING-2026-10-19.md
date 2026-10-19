# EXP-053 — human-operated offline signer key-read security finding

**October 19, 2026. NO_GO_RESEARCH_ONLY. This is an automated source-only
negative test, not independent custody, enrollment or an actual human signature.**

## Reproduction

The existing `exp053_sign_review.get_key(path)` used `path.stat()` and later
`path.read_bytes()`. Both follow a symlink, and they do not bind the permission
check to the file contents through one open file descriptor. A disposable
synthetic Ed25519 PKCS8 PEM stored mode 0600 was accepted when loaded via an
attacker-controlled final-component symlink. Real reviewer keys were not read,
created or enrolled and this is not proof that any live key was compromised.

## Corrected boundary

`get_key()` now requires POSIX `O_NOFOLLOW` and `O_NONBLOCK` before opening.
It reads <=16 KiB+1 bytes from the SAME non-following descriptor after checking
owner uid, private mode, regular-file type, single-link count and bounded
non-empty size. `fstat()` before/after must agree on device, inode, size,
mtime and ctime; identity drift or short reads are HOLD. The descriptor is
closed on every read/parse outcome. Only a valid unencrypted Ed25519 PEM is
returned; invalid material is rejected without printing paths or key data.

Synthetic regression tests cover owner-private successful public-key
extraction, last-component symlink, world-readable file, hardlink, FIFO,
invalid/wrong-algorithm PEM, missing secure OS flags, oversize/empty file,
read failure and rename/replace races. They use only ephemeral unit-test keys,
no reviewer identities or legal RACE passages. A filename swap of the opened
key caused a ctime change and thus a fail-closed HOLD; the tool did NOT sign
using a replacement. It is a bounded point-in-time check: it does not defend
against an attacker with the same user's process and key privileges, symlinks
in all ancestor directories or external custody corruption.

## Execution/review limits

This patch does not sign any review, create a human-identity/consent record,
verify independent key control, pin a reviewer roster, admit a single RACE
article, deploy services or run inference. Real reviewers must personally
control their signing keys and remain blind to the sealed answer key and each
other's responses. The study owner must independently vet and enroll them.
The source-only review handoff must itself undergo accountable human review
before distribution. Zero human reviews remain admitted and all scientific
and independent witness approvals remain NO-GO.

Reproduce with `python3 -m pytest -q tests/test_exp053_signing_key_read.py`.
