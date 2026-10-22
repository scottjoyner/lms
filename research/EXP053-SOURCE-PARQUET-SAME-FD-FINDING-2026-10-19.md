# EXP-053 packet construction source SHA/Parquet path substitution

**October 19, 2026. Synthetic-only evidence. NO_GO_RESEARCH_ONLY.**

## Reproducible weakness

Previously `corpus()` verified `filehash(SOURCE)` and then separately gave
`SOURCE` *as a pathname* to `pyarrow.parquet.read_table`. The actual
registered Hugging Face snapshot path's last component is a symlink (confirmed
through **metadata only**, no corpus content read). With an invented, 18-byte
"source A", distinct invented "source B", a synthetic snapshot symlink,
and a test-only `pyarrow.parquet` stand-in, swapping the symlink immediately
after A was hashed made the old builder return **2,500 synthetic rows from B
under A's verified SHA**. No licensed passage, exam answer or actual human
reviewer was used. This affects *corpus-to-blinded-packet construction*, not
the separately pinned and signed scientific admission logic.

## Source-only repair

- Fail closed for a missing/changed registered protocol SHA or malformed
  source SHA before accessing the Parquet payload.
- Preserve genuine static Hugging Face snapshot symlinks by resolving the
  lexical `SOURCE` to a target path once and securely opening that target
  through the existing nonfollowing per-ancestor descriptor walk.
- Require nonempty regular single-link source <=64 MB. SHA-256 the exact
  source bytes through one open file object, rewind **that same descriptor**
  and pass the object, rather than a path, to `pq.read_table`.
- Before returning rows, refuse changed dev/inode/size/mtime/ctime,
  a changed snapshot symlink target, or a different final resolved inode.
- Deny on unavailable secure OS flags, Parquet reader missing, FIFO, source
  overrun, mismatched pinned SHA, and insufficient row count. Never log text.

Eleven synthetic regression tests cover a stable snapshot symlink, exact
source digest, symlink swap during decode, same-byte inode replacement,
mid-decode source rewrite, invalid pin, unavailable pyarrow, missing kernel
flags, oversized source, FIFO and preregistered protocol mismatch. The Fleet
Commander research host intentionally lacks `pyarrow`: a stand-in module
validates file-descriptor binding and fail-closed control flow, **but does not
validate actual PyArrow native file-object compatibility or real Parquet
parsing**. That requires a separately authorized, license-safe integration
witness with a real PyArrow installation before any live packet rebuild.

No scientific cohort rerun, private RACE passage retrieval, review signature,
model forward, inference authorization, repo merge or service deployment was
performed. The real owner reviewer pin and weight approval pin remain unset.
Independent human code review and external custody/anti-rollback evidence
are not replaced by these local source tests.

## Verification

`python3 -m pytest -q tests/test_exp053_source_parquet_same_fd.py`

**Decision: NO_GO_RESEARCH_ONLY.**

## Supplemental native PyArrow cross-check on xwing

After the original finding, a system-wide read-only package survey found
PyArrow 25.0.0 installed on xwing. Using ONLY 2500 invented Parquet rows in
an automatically deleted /tmp directory, the first native threaded
`pq.read_table(source)` correctly decoded the rows but the interpreter
subsequently ABORTED at shutdown (exit 255, `terminate called without an
active exception`). This is not a passing runtime test. A non-threaded
`pq.read_table(source, use_threads=False)` decoded 2500 rows with clean
shutdown. An additional three separate interpreter processes repeated the
exact chained native `read_table(..., use_threads=False).to_pylist()` call
and all exited 0 with 2500 rows, a stable file descriptor and symlink
pointer. The code now explicitly uses `use_threads=False`. Its source-only
mock tests fail if threading is reenabled. See the separate frozen native
smoke findings document. These synthetic checks do not validate the exact
licensed corpus, a real human receipt, or deployment on the production
research host. Independent code review and scientific NO-GO remain.
