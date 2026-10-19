# EXP-053 — atomic pinned public manifest read

**Research-only specification, frozen before implementation (October 19, 2026).**
**NO_GO_RESEARCH_ONLY**: no owner-approved roster, human review, independent
custody or model forward is established by this work.

## Reproduced public-safe integrity defect

`verified_admission` hashed the preregistered, source-free manifest using
`filehash(MANIFEST)`, then separately parsed `MANIFEST.read_text()`. Under a
synthetic exact-timing replacement between these two operations, the
independently pinned digest represented the ORIGINAL manifest but the
`packet_sha256` handed to `load_roster` came from a DIFFERENT manifest.
The experiment stopped before accessing licensed RACE data or private forms.
Even if a separately pinned private roster makes a full exploit harder,
trust-anchor hash/read disagreement must never be silently accepted.

## Exact source-only correction

- Keep the early missing/invalid owner-pin HOLD before **any** filesystem
  source/manifest activity.
- Open the immutable public manifest by the same non-following component-
  anchored directory traversal as the private reader. Require Linux
  `O_PATH|O_DIRECTORY|O_NOFOLLOW|O_NONBLOCK` or HOLD. Final fd must represent
  a single-linked regular file; bound bytes to <= 1 MiB.
- Read bytes using one fd and verify that its fstat dev, ino, size, mtime_ns
  and ctime_ns remain stable. Check the *same raw bytes* against the
  preregistered `MANIFEST_SHA` **before parsing**; SHA must be canonical
  lowercase hex. Then parse the exact buffered bytes as strict UTF-8 JSON
  with no duplicate keys, NaN, Infinity, numeric overflow or nonobject root.
- Re-resolve and securely reopen final lexical manifest path, requiring
  unchanged path identity and same underlying inode. Close all descriptors.
  Refuse any symlinked ancestor, swapped final component, hardlink, FIFO,
  oversized file, incomplete read or invalid JSON with sanitized HOLD.
- Retain source SHA/protocol SHA, packet identity, sealed-answer binding,
  pinned independently vetted reviewer roster and 40-item human threshold.
- Update tests that instrument `filehash(MANIFEST)` to recognize that
  the manifest digest is no longer acquired by a separate hash call. Add
  synthetic tests for the original hash/read substitution, altered raw SHA,
  alternate file and symlink, malformed JSON and exact legitimate manifest.
  No real private packet, RACE passage, signer key or weight file may be read.

A bounded POSIX fd read is not proof of protection against privileged or
same-user post-validation directory swaps. Independent custody and
cryptographic checkpoints remain separate hard gates. No merge/deployment.
