# EXP-053 — independent technical review of score JSON numeric fidelity

DRAFT source-only patch stacked on #45 (and prior #44→#37→#32).

Please review `exp053_execution_attestation.unambiguous_json`, specifically
binary64 finite range and loss of nonzero decimal value through underflow
before a signed score archive is analyzed. Confirm the parser does not reject
true-zero mantissas only because of exponent digits, accepts valid finite
and representable subnormal scores, and never exposes private values on
error. Confirm attestation, operator and retry-log callers still enforce
their field schemas and that the independent signed-key gates remain empty.

Synthetic-only checks:

```
python3 -m pytest -q tests/test_exp053_scorer_numeric_json_fidelity.py
python3 -m pytest -q tests/test_exp053_*.py
python3 -m ruff check research/exp053_execution_attestation.py \
  tests/test_exp053_scorer_numeric_json_fidelity.py
```

No real operator key, external witness service, model checkpoint, model
inference, genuine signed human review or source article opened. Keep all
PRs draft, do not merge/deploy or set owner approval pins. External qualified
technical review is distinct from independently blinded RACE annotations.
**NO_GO_RESEARCH_ONLY.**
