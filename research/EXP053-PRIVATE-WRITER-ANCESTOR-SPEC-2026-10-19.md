# EXP-053 — private artifact output ancestor provenance

**Frozen source-only security test specification, 2026-10-19.**
**NO_GO_RESEARCH_ONLY**. No real reviewer, licensed source, or model output.

## Synthetic reproduction

`save_private` had O_EXCL|O_NOFOLLOW on the final leaf and O_DIRECTORY|
O_NOFOLLOW on the immediate parent, but `os.open(path.parent)` followed
*earlier* directory symlinks. A disposable mode-0700 directory under `/tmp`
with a synthetic symlink used as the parent alias successfully received a
new mode-0600 invented review JSON. Leaf mode does not fix wrong-directory
custody: signed review artifacts could be silently written via a pathname
with a substituted intermediate ancestor.

## Required correction and tests

- Require Linux O_PATH, O_DIRECTORY, O_NOFOLLOW, O_NONBLOCK and fail closed
  when unavailable. Use `exp053_reviewer_roster._open_no_symlink_ancestors`
  to open BOTH the initial and rechecked parent directory file descriptor.
  That shared helper opens EVERY lexical ancestor with O_PATH, not just the
  final parent path. Require that the resulting parent fd represents a
  directory owned by effective uid; reject non-directories and missing uid
  control before creating the output leaf.
- Preserve current O_EXCL|O_NOFOLLOW|O_NONBLOCK leaf creation with mode 0600
  from birth, partial-write handling, fsync, readback, inode and parent
  identity checks, and inode-bound cleanup of interrupted writes. Never
  unlink an unrelated replacement filename if the original inode moved.
- Add synthetic tests showing final parent symlink and ancestor symlink
  produce HOLD and leave no output; all secure OS flags must be present;
  a parent with a faked wrong uid must deny before any output creation;
  a parent swapped to an ancestor symlink during a write must deny without
  touching a decoy or leaving partial output. Preserve old positive writer
  tests and signer integration tests using disposable ephemeral keys only.
- Do not change build packet schema, public manifest bytes, owner pins,
  source SHA, review signatures, or admission thresholds. No licensed
  corpus build, real signer export, release, merge, or inference.

The path revalidation only protects against detected pathname races and
cannot defeat an adversary sharing the same uid after the final check.
Independent high-water checkpoints, custody roots, real human reviewers and
model-weight authorization remain unsatisfied. NO_GO_RESEARCH_ONLY.
