# EXP-053 — source-only technical review: aggregate destination binding

This draft patch is stacked on PR #42, in the same human/weight NO-GO chain
#41 → #40 → #39 → #38 → #37 → #32. Review only fake aggregate JSON and
synthetic directory / symlink test cases.

The change adds an opt-in required parent inode to `save_private` without
changing existing packet and signer callers. It validates the intended
research directory BEFORE consuming a real operator claim and demands the
same inode when creating the final aggregate AFTER the claim. Review the
pre-claim/post-claim ordering, O_EXCL and mode-0600-from-birth behavior,
symlink and inode substitutions, inode-bound cleanup, syscall error modes,
and the deliberate fail-closed irreversibility of the operator claim if
publication fails.

```
python3 -m pytest -q tests/test_exp053_aggregate_output_dirfd.py
python3 -m pytest -q tests/test_exp053_*.py
python3 -m ruff check research/exp053_race_review.py \
    research/exp053_verified_analysis.py \
    tests/test_exp053_aggregate_output_dirfd.py
```

The earlier output was formatted human-readable via `json.dumps(indent=2)`.
This source-only revision uses the canonical compact JSON serializer already
used for the audited writer. Public aggregate keys and data remain the
same; the output is initially **owner-only mode 0600** and requires a
separate approved publication step to become readable publicly. Neither
formatting nor CI validates that actual model execution occurred.

No real blinded human review, operator claim, model weight, RACE corpus,
scoring record, signed scientific evidence or production output is touched.
Independent human and operator custody, review approvals and the minimum
40 adjudicated articles remain blocked. **NO_GO_RESEARCH_ONLY.**
