# EXP-053 — independent technical review: same source fd for packet building

Draft stacked review branch based on the exact source of PR #38
(`68ffb084cec264330c488482ae447b4de063f697`).
The previous source builder trusted a hash of `SOURCE` but decoded a
*different pathname resolution* after its Hugging Face snapshot symlink
changed. A synthetic fixture proved this. The source now opens the resolved
source blob once, SHA-hashes then rewinds then decodes using **one fd**,
and rechecks pathname and inode identity. This preserves a stable snapshot
symlink instead of denying every symlink automatically.

Independent review request: examine symlink resolution before/after reads,
per-component no-follow semantics, the size limit, stability fstat checks,
source exception/refusal behavior and whether `pyarrow.parquet.read_table`
can decode the supplied open Python binary file object on target installations.
The current x1-370 environment has NO `pyarrow` installed and the tests
intentionally use a fake reader and **synthetic invented source bytes**.
Native-parquet compatibility is therefore separately unverified, not a green
or an inference entitlement. Do not download or expose actual RACE passages
for this technical review.

Run without licensed source or optional pyarrow:

```
python3 -m pytest -q tests/test_exp053_source_parquet_same_fd.py
python3 -m pytest -q tests/test_exp053_*.py
python3 -m ruff check research/exp053_race_review.py \
  tests/test_exp053_source_parquet_same_fd.py
```

Only source, test and read-only technical documentation belong in this PR.
Do NOT merge, deploy, launch actual RACE decoding, set signer approval pins,
claim human independence, issue scorer approval or run model forwards.
**NO_GO_RESEARCH_ONLY**; at least 40 independently co-reviewed articles,
external witness operator/high-water control and owner model authorization
remain unsatisfied.
