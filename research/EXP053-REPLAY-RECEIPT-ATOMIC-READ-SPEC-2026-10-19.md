# EXP-053 — scoring replay pinned metadata atomic reader

**Frozen synthetic-only technical spec (2026-10-19). NO_GO_RESEARCH_ONLY.**

## Reproduced defect

The verified scorer `admitted_weight_inventory` computed the independently
pinned SHA-256 of `WEIGHT_REPORT`, then separately parsed a fresh
`WEIGHT_REPORT.read_text()`. After hashing synthetic report A, a test swapped
in a *different*, internally consistent invented report B. The old checker
returned a model receipt from B, despite B not matching the pinned A bytes.
The actual owner WEIGHT_REPORT_SHA is intentionally unset, so this is not
proof that production approval was bypassed. No real weights opened.

Also `require_sources` hashed pinned source-free tokenizer preflight and public
cohort manifest separately from JSON parsing and read its private packet via
a raw path-following read. It must reuse the hardened source-free manifest
and protected packet readers introduced in source-only draft PR #38.

## Required behavior

- Never open a pinned weight report while WEIGHT_REPORT_SHA is unset or
  malformed; retain the independent operator requirement and all checkpoint
  schema/individual receipt/filename controls.
- New `_read_pinned_source_free_receipt(path,pin)` must require canonical
  lowercase hex SHA, nonfollowing Linux O_PATH ancestor and leaf access,
  regular single-link file, nonzero size <=1 MiB, stable dev/ino/size/
  mtime_ns/ctime_ns before/after bounded single-fd read, hash of exactly
  those bytes, stable resolved pathname and final reopened inode. JSON
  parsing of buffered UTF-8 must reject duplicate keys, NaN/Infinity,
  numeric overflow, malformed content, and non-dict root. All mismatch
  errors produce sanitized `HOLD EXP053` with no private content logging.
- `admitted_weight_inventory` must pass this parsed **same-bytes** report
  into all existing checkpoint receipt checks.
- `require_sources` must use this helper for the public tokenizer receipt;
  `_read_pinned_public_manifest()` for the exact immutable public cohort
  manifest and `_read_private_admission_json()` for the caller-selected
  private packet. Do not silently fall back to the default private packet.
- Synthetic regressions must show hash-then-swap cannot return model B,
  duplicate and nonfinite JSON rejection even with a synthetically pinned
  digest, oversized, symlinked, hardlinked files and missing secure flags;
  source-free positive receipt and no approval-pin early HOLD. Use only
  disposable fake weight metadata, not any model files, RACE passages,
  private signer material or real scientific scores.
- Preserve the complete PR #37..#41 chain and source-only authorization
  requirements. Do not merge/deploy/run inference or set roster/weight pins.

This is code hygiene and immutable-byte trust binding, not true external
human, witness, or model execution authorization. NO_GO_RESEARCH_ONLY.
