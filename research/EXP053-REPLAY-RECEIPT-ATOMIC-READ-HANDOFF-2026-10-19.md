# EXP-053 — stacked draft code review: same-byte source-free scorer receipts

Stacked on PR #41 (`0de5117178581310e697b2a8e05228f75a56f2a3`),
which is stacked on security PRs #40 → #39 → #38 → #37 → #32.

Review the public-source checkpoint identity receipt's pinned SHA and JSON
parsing via one stable descriptor, preservation of fail-closed empty owner
weight pin, and refusal of any swapped/malformed file. Verify the same
source-free pinned reader for tokenizer metadata, the older atomic immutable
public cohort manifest check, and guarded owner-private packet reads.
Check Linux O_PATH/O_NOFOLLOW portability and same-UID post-check limitations.

Tests only involve disposable artificial model shard SHA strings and source-
free public receipts; no licensed article, neural model weight, actual human
reviewer key, model inference or scientific score is opened or published.

```
python3 -m pytest -q tests/test_exp053_verified_replay_atomic_receipts.py
python3 -m pytest -q tests/test_exp053_*.py
python3 -m ruff check research/exp053_verified_analysis.py \
  tests/test_exp053_verified_replay_atomic_receipts.py
```

An independently trusted reviewer may evaluate code, but code review is NOT
human blinded scientific annotation, attested model execution or operator
signoff. Both `APPROVED_REVIEWER_ROSTER_SHA256` and `WEIGHT_REPORT_SHA` remain
empty; >=40 signed/adjudicated unique articles and off-host witness custody
are still missing. Both this PR and its parents must remain draft and unmerged.
**NO_GO_RESEARCH_ONLY.**
