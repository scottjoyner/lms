# EXP-053 private review artifact exclusive-creation safety

**Research-only; NO_GO_RESEARCH_ONLY. October 19, 2026.**

A real synthetic test demonstrated that the previous `save_private()` created
files with `open(path, "xb")` at the process's default creation mode, commonly
`0666 & ~umask` = `0644`, *then* applied `fchmod(0600)`. Interrupting the
`fchmod()` call left an owner-uncontrolled `0644` file. The same sequence
existed in the offline human signer `sign()` output. On directory trees with
permissive parent modes, this can expose unpublished, licensed passage data
or human review material. No private research content was read for this test.

Required narrow fix, before implementation:

- Use one shared source-only exclusive writer for research packet, sealed key,
  unsigned forms, signed human responses, and admission receipts.
- Open final file with `os.open(... O_CREAT|O_EXCL|O_NOFOLLOW ..., 0o600)`
  relative to a non-following directory file descriptor. Refuse missing
  `O_NOFOLLOW`/`O_DIRECTORY`/`O_NONBLOCK`. Do not create temporary public files
  or use path-following, permissive-creation `Path.open("x[b]")`.
- Validate file ownership, regular-file type, link count and private mode
  BEFORE any data is written. Apply `fchmod(fd, 0600)` only after creation,
  where the initial mode already forbids group/other access.
- Write all bytes including the canonical newline through that exact fd;
  fsync, bounded readback, final-entry inode and content identity verification
  before returning success. Preserve exclusive no-overwrite semantics.
- On ordinary exceptions or keyboard interruptions after creation, close
  descriptors and best-effort remove only the entry whose inode matches the
  created file, relative to the original opened parent fd. Do NOT delete a
  separately substituted file; if identity cannot be established refuse and
  leave any uncertain artifact blocked. No guarantee on power-loss/SIGKILL,
  same-UID attackers after the final check, or ancestors of a trusted parent.
- Use tests on entirely synthetic tiny JSON under a disposable filesystem,
  including normal `022` umask, failure before fchmod, failure during write
  and fsync, partial write, symlink target, existing-file preservation, parent
  and final-entry substitution, and signed export before/after key access.
- This hardening does NOT authorize human reviewers, modify source manifests,
  introduce production services, or allow EXP-053 inference. Exact frozen
  candidate study and consent/identity/custody gates remain intact.
