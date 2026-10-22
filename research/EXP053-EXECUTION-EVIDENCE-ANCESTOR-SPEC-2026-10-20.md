# EXP-053 — private execution evidence: no symlink ancestors

**Frozen before implementing change, 2026-10-20. NO_GO_RESEARCH_ONLY.**

The existing execution-attestation helper `private_regular_file` checks
whether the resolved evidence path lies within a resolved private directory
and rejects a symlink final leaf, but an intermediate symlink inside the
private directory can still redirect the opened evidence pathname. In a
synthetic tempdir test it returned invented private JSON through a linked
intermediate directory. It also used `getattr(os,"O_NOFOLLOW",0)`, silently
removing the final-component protection if a Linux flag was unavailable.

## Required correction

- Preserve same-FD incremental SHA and optional bounded JSON bytes behavior.
- Reject path traversals outside the literal, absolute selected private root,
  without following symlinks. Reject path equal to private root.
- Fail closed if O_PATH, O_DIRECTORY, O_NOFOLLOW, O_NONBLOCK unavailable.
  Use the already source-reviewed `_open_no_symlink_ancestors` for every
  directory component and evidence leaf. Require regular, single-linked,
  owner-UID file with private mode (no group/other permission).
- Require stable descriptor dev/ino/size/mtime_ns/ctime_ns before/after read,
  exact size count and re-open the *lexical path* under same no-follow rules
  after hashing to detect inode/path substitution. No data is logged.
- Only owner-private, invented evidence bytes in synthetic tests. Prove
  static ancestor symlink refusal, replaced path during read refusal,
  same-byte inode substitution refusal, missing O_NOFOLLOW/O_PATH controls
  fail closed, final symlink/FIFO/hardlink/wrong-uid/weak-mode denial, clean
  owner-only success and no change to existing execution attestation tests.
- Change only the shared evidence reader, its focused tests, spec/finding
  and review handoff. Do not edit real scorer data, publish model metrics,
  enroll keys/reviewers, run licensed RACE or models, merge or deploy.

No filesystem-only validation is an external, tamper-resistant witness;
privileged/same-UID modifications after the final check remain a threat.
All independent human, operator, witness and weight pins remain unset.
**NO_GO_RESEARCH_ONLY.**
