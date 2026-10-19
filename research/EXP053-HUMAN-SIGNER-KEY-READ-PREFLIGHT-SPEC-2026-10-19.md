# EXP-053 offline human signer key read security specification

**2026-10-19. Research only; NO_GO_RESEARCH_ONLY.** This document describes
only the private-key reader for a human-operated OFFLINE review signing tool.
An agent MUST NOT generate a real reviewer key, sign any real review,
authorize reviewer enrollment, execute models, or claim an independent human.

Reproduced defect: `exp053_sign_review.get_key()` performs `path.stat()`
(which follows symlinks) and then `path.read_bytes()` (another following path
lookup). A disposable unit test with an entirely synthetic Ed25519 PEM showed
that a symlink to a mode-0600 key was accepted, despite the key path not being
a private regular file. The separate path reads also allow source replacement
between checking permissions and obtaining the bytes.

The proposed source-only correction must:

- Require POSIX `O_NOFOLLOW` and `O_NONBLOCK` and refuse to degrade silently.
  Open the final key file with `os.open(O_RDONLY|O_NOFOLLOW|O_NONBLOCK)` and
  read bytes from that SAME descriptor with `os.read` (never read via a second
  pathname). The application does not make claims about symlinks in ancestor
  directories or adversaries controlling the same user account.
- Require the fd to reference a regular file, owned by the effective user,
  with no group/other bits, exactly one link, and a bounded positive key
  file size <= 16 KiB. Permissions are checked before reading key material.
- Read <= 16 KiB+1 bytes; reject short, too-long or changed reads. Compare
  device, inode, size, mtime and ctime before/after reading. Close the fd
  on ALL errors. Never print, trace or export private-key bytes.
- Load only unencrypted Ed25519 PEM keys using the preexisting cryptography
  API; reject wrong format, other algorithms and malformed material with a
  sanitized `HOLD` error that contains neither filesystem path nor bytes.
- Preserve the existing CLI, signing workflow, output exclusivity, and public
  key semantics. Add synthetic, disposable tests for normal Ed25519 public
  derivation, symlink refusal, world-readable/hardlinked/nonregular files,
  missing safety flags, size bounds and read failure. No real reviewer keys,
  user identities, RACE passages, or signing approvals in fixtures.

The existence of an owner-private key file does NOT establish an independently
verified human identity, independent key custody, blind review, external
witness durability or eligibility for scientific inference. All owner pins
remain blank and EXP-053 model-forward execution remains on HOLD.
