# EXP-053 operator journal and used-marker ancestor custody — frozen test contract

**2026-10-20. Source/test/docs only. NO_GO_RESEARCH_ONLY.**

## Reproduced synthetic weakness

The single-use claim verifier checked the final `path.is_symlink()` and used
`O_NOFOLLOW` (when supported) for journal open and used-marker creation, but
opened journal/marker paths by name without protecting **ancestor** directory
symlinks. A disposable synthetic operator claim, with generated ephemeral
keys and a mock same-process witness CAS (NOT a trusted live authority),
accepted a journal reached via an untrusted symlinked parent and advanced
its state/marker. The code used `getattr(O_NOFOLLOW,0)` so a missing flag
could silently remove the very control the caller expected.

## Exact source-only repair

- Fail closed unless O_PATH, O_DIRECTORY, O_NOFOLLOW and O_NONBLOCK exist.
  Open the lexical operator journal directory with the existing component-
  by-component O_PATH/O_NOFOLLOW walker; require an owned, non-group-writable,
  non-world-writable directory. Open the **journal leaf** with O_NOFOLLOW via
  the checked parent dirfd; require regular, owner-only, single-linked inode.
- Lock the opened journal fd as before. Check the opened inode against
  `os.stat(leaf,dir_fd=parent_fd,follow_symlinks=False)` rather than a fresh
  absolute-path lookup.
- Reopen the lexical parent with the same non-following walk and compare
  dev/ino to the original pinned directory BEFORE any irreversible remote
  CAS, AFTER the CAS but BEFORE creating the local marker, and BEFORE
  returning success. If the path moved, HOLD and do not redirect writes.
- Create the used-execution marker (`O_CREAT|O_EXCL|O_NOFOLLOW`, mode 0600)
  with `dir_fd=parent_fd` and fsync the *same pinned directory fd*, never a
  separately resolved absolute directory. Journal fd and parent dirfd must
  be closed on all paths. A post-CAS failure must NEVER automatically reset
  the remote witness, local state, or marker; preserve fail-closed semantics.
- Preserve the exact signed authorization, external witness request/receipt
  verification, sequence ownership, operator/scorer separation, replay
  rejection and output format; all real authority roots stay empty.
- Regression with invented data ONLY: static symlinked parent refuses before
  fake CAS; missing safe flags refuse before file open; parent substituted
  during fake CAS refuses without writing a marker into the new directory;
  same-owner replacement inode refuses; successful normal claim still
  writes one marker and advances journal; repeated execution and concurrent
  callers remain fail closed. NEVER use real human source, keys, weights,
  score data or a real external CAS endpoint.

Same-UID adversaries and privileged swaps after the last pathname check are
not prevented by fd-pinning alone. Independent physical operator/key custody,
rollback-resistant external high-water service, qualified technical reviewers,
40 genuine blinded co-reviewed/adjudicated articles and weight authorization
remain unverified, irrespective of synthetic test success. NO_GO_RESEARCH_ONLY.
