# EXP-053 source-free parser fail-closed remediation — 2026-10-18

**Research-only, NO_GO_RESEARCH_ONLY.** This is defensive input handling for
existing controller-local tools, not a new custody, signature, or external
checkpoint authority. All actual EXP-053 execution and witness deployment
remain blocked on independent administrative custody and adjudication.

## Concrete failure mode

Although incoming JSON was byte-bounded (16–48 KiB), a recursively nested
array within those limits could trigger `RecursionError` in the Python JSON
parser, or during canonical reserialization of a deeply nested object. Before
this patch, shared source-free parsers could propagate that exception as an
uncaught traceback through downstream command-line tools, defeating their
contract to return a sanitized failure status. Such input must be treated as
malformed, not as a programmer exception or partial approval.

## Scope of remediation

- `exp053_custody_boundary_audit.parse` maps parser recursion exhaustion to
  the existing `Hold` denial instead of raising an uncaught exception.
- `exp053_custody_receipt_consistency._strict_load` maps JSON recursion
  exhaustion to `Hold` and `_canonical` likewise maps recursive/nonserializable
  data to `Hold`. This guards the common helpers used by downstream receipt,
  cross-bundle, checkpoint-comparison and observer-publication chain modules.
- `tests/test_exp053_nested_json_failclosed.py` covers extreme nesting,
  recursively nested objects during canonicalization, and bounded local CLI
  invocations with pathological claims, bundle, reference and publication
  files. Its adversarial inputs are synthetic. Test-only observer keys, where
  needed, are ephemeral and never installed as a trusted root.

No frozen scientific, custody or signing specification was changed. No gate
was changed to allow execution. All CLI failures remain nonzero status 2 with
source-free JSON, not raw filenames, stack traces, keys, private evidence,
research passages, or account information. All existing NO-GO controls and
unconfigured witness/operator/runner signer pins remain intact.

## Minimal reproducibility

```
python3 -m pytest -q tests/test_exp053_nested_json_failclosed.py
python3 -m pytest -q tests/test_exp053_*.py
python3 -m ruff check research/exp053_custody_boundary_audit.py \\
  research/exp053_custody_receipt_consistency.py \\
  tests/test_exp053_nested_json_failclosed.py
```

The only material milestone in this patch is reliable sanitized refusal for
malformed input. Separate external, independently pinned and published
high-water heads, true admin/key separation, physical privilege denial tests,
and >=40 independent RACE adjudications remain unsatisfied. No fleet hosts,
model inference, NAS/Neo4j, GitHub release or production services are involved.
