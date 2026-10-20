# EXP-053 — independent review target: execution evidence path provenance

Source-only draft technical review stacked on draft #44, itself on #43 through
#32. No licensed corpus, real independent signer, approved model runner,
operator high-water service or real model scores are included.

Review the source-only change to `private_regular_file` in
`exp053_execution_attestation.py`. Review safe Linux flag fail-closed
behavior, lexical private-root relationship, per-directory O_PATH/O_NOFOLLOW
traversal, regular-single-link owner-private file checks, same-FD hash and
bounded JSON retention, inode/path re-opening, exception cleanup and
syscall error paths. Verify that score/archive/attempt-log and operator
signed-authorization callers still behave correctly and that no actual
approved model evidence is required to exercise these tests.

Run synthetic-only local checks:

```
python3 -m pytest -q tests/test_exp053_execution_evidence_no_symlink_ancestors.py
python3 -m pytest -q tests/test_exp053_*.py
python3 -m ruff check research/exp053_execution_attestation.py \
  tests/test_exp053_execution_evidence_no_symlink_ancestors.py
```

Code-level technical approval is separate from two independent blinded
human scientific reviewers, at least 40 co-reviewed/adjudicated article
acceptances, independently administered operator/witness root custody, and
owner-approved model identities. `NO_GO_RESEARCH_ONLY`. Do not merge,
deploy, set approval SHA pins, create real signer receipts or run inference.
