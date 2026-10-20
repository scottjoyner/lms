# EXP-053 — draft source-only review: complete sealed case scope

Stacked on draft PR #40 (`8ed7a6b06b33efc625acae2f3068805c6986b865`).
The source-only change adds structural validation to the already digest-
and-signature-gated research admission path; it does NOT enroll real humans
or access licensed RACE source.

Review `_validate_sealed_case_scope`, its call point before A/B reviewer
private-file reads, and the synthetic tests. Explicitly assess: exact
64-case cohort vs minimum 40 accepted threshold, duplicate/foreign/missing
cases, canonical answer labels, malformed container/manifest denial, and
whether every sealed answer maps to exactly one registered case. Verify
that the preregistered manifest SHA and owner-approved roster digest remain
independently mandatory before the trusted seal can be read.

Run only source-free fixtures:

```
python3 -m pytest -q tests/test_exp053_sealed_case_scope.py
python3 -m pytest -q tests/test_exp053_*.py
python3 -m ruff check research/exp053_race_review.py \
  tests/test_exp053_sealed_case_scope.py
```

The historical reproduction **mocked all real trust roots and signatures**;
it does not prove actual human evidence was admitted. Both human roster
SHA and model-weight SHA remain unset. All related PRs remain draft, with
no human review, merged release, live packet rebuild or model inference.
**Scientific decision: NO_GO_RESEARCH_ONLY.**
