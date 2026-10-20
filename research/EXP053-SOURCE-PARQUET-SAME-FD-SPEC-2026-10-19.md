# EXP-053 — source parquet hash/decode same-fd provenance contract

**2026-10-19, preregistered synthetic-only technical test specification.**
**Research remains NO_GO_RESEARCH_ONLY; no real signer or model inference.**

## Reproduced bug

`corpus()` first verified `filehash(SOURCE)==SOURCE_SHA`, then called
`pq.read_table(SOURCE)` by path. The real Hugging Face snapshot source path
has a **symlink final component**. With disposable synthetic Parquet stand-in
files A/B and a test-only `pyarrow.parquet` module, changing the snapshot
symlink after hashing A but before `read_table(SOURCE)` caused the old code
to accept 2500 invented rows read from B under A's SHA-256.
This is the *packet construction* path (not the signed-review admission path).
This test accessed NO licensed RACE rows or real reviewer material.

## Exact fix and negative tests

1. Verify the pinned preregistered protocol SHA before accessing source
   contents; require source SHA to have exact lowercase 64-hex syntax.
2. Preserve legitimate Hugging Face snapshot final symlinks: resolve the
   source pathname once with `strict=True`, then open its resolved target
   through the per-ancestor O_PATH/O_DIRECTORY/O_NOFOLLOW walk and an
   O_NOFOLLOW/O_NONBLOCK final leaf. A static symlink is permitted only if
   its resolved target and inode remain stable throughout.
3. Require a regular nonempty single-link source file up to 64,000,000 bytes
   (real pinned source is ~1.7 MB). Read and hash exact source bytes through
   **one open file descriptor**. `seek(0)`, then supply the same **open file
   object** to `pyarrow.parquet.read_table`, never re-enter via path.
   On completion compare same-FD fstat identity (dev, ino, size, mtime_ns,
   ctime_ns), verify original source lexical pathname still resolves to same
   target, and compare a securely reopened target dev/ino before returning.
4. Deny with sanitized HOLD if symlink target changes at any phase, source
   file changes during hashing/Parquet read, source does not match pinned SHA,
   wrong file kind/size, non-follow file-access flags unavailable, or
   pyarrow backend missing. Do not print contents or read a second path.
5. Tests use *only* invented small files and a synthetic Parquet stand-in
   module (the research host does not have pyarrow); assert same open
   file object reaches the reader, return 2500 invented rows, reject
   symlink swap, inode swap, content changed during read, missing/incorrect
   SHA, absent backend, oversized source, FIFO, and missing O_PATH flags.
   Include a positive stable snapshot-symlink case.
6. No change to registered cohort/packet schema, real source SHA,
   external license terms, private human-review pin, model-weight pin,
   adjudication requirements, or downstream signed admission.

This is code-level TOCTOU protection, not a guarantee against a privileged or
same-user adversary after final validation. It does not establish independent
witness custody and MUST NOT authorize any model forward, scientific
admission, merge or deployment by itself.
