# EXP-053 roster directory descriptor cleanup contract

2026-10-19. Research-only; NO_GO_RESEARCH_ONLY. Frozen before code edits.

The recently introduced nonfollowing `O_PATH` directory walk was tested with
synthetic injected `os.close` failure. The helper opened a replacement
directory descriptor, attempted to close the previous one, and left the new
fd live when the previous close raised. A separate failure closing the final
parent can likewise leave a successful final-file descriptor live after the
function refuses. These are resource-ownership bugs during I/O failures, not
evidence of authority bypass or actual review material access.

Requirements:

- If the old parent fails to close during a handoff, attempt to close the
  newly acquired `next_dir` fd; the outer finally still closes the original
  parent. Do not proceed into the roster file, return a fd, or suppress the
  failed operation as a successful read.
- If the final parent fails to close after the leaf fd opens, attempt to close
  the leaf too. The parent may have been closed by the OS despite the reported
  error, so never blindly retry: first probe the fd and require that its
  device/inode match the identity captured before close, then best-effort
  release the still-open parent. No opened leaf may escape on failure.
- Preserve the per-component no-follow `O_PATH|O_DIRECTORY|O_NOFOLLOW` walk,
  the leaf `O_RDONLY|O_NOFOLLOW|O_NONBLOCK` flags, bounded roster bytes,
  independent owner pin check, SHA binding, and public repo exclusion.
- Synthetic-only tests inject an `OSError` in intermediate/final close calls
  and inspect all captured descriptors with `os.fstat` afterward. Test success
  normally and refuse/cleanup on interruption without opening licensed RACE
  files, owner signing keys or a real reviewer roster.

This is not proof of resistance to SIGKILL, kernel-close ambiguity, same-UID
adversaries, independently signed observations, or completed human review.
No approval pin, production workload or scientific execution may be changed.
