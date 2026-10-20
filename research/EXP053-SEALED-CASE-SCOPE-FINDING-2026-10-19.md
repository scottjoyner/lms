# EXP-053 — sealed answer-key scope and source-packet completeness

**2026-10-19. Synthetic-only admission-contract finding. NO_GO_RESEARCH_ONLY.**

`verified_admission` originally verified a sealed answer-key digest against
the preregistered manifest, but then iterated *only the keys in that sealed
answer map* to construct the accepted/rejected/unresolved receipt. It did
not independently verify that the sealed keys cover **all** of the packet's
64 distinct registered cases. To demonstrate the missing contract, a test
constructed 64 entirely invented case records and 40 invented answer labels,
then **mocked the independently pinned immutable manifest, reviewer roster,
source file hashes, signer verification and private file reads**. The old
algorithm returned a synthetic `human_approval=True` structure with 40
accepted entries even though the input packet held 64 cases. The deliberate
bypass of real trust roots makes this a **structural consistency bug, not
proof of a live forged human approval** or compromise of the real pinned
public manifest. No actual licensed RACE passage or real reviewer receipt
was opened.

## Repair

After verifying exact committed packet SHA, the new
`_validate_sealed_case_scope` checks that:

- the public manifest carries a canonical lowercase sealed answer-key SHA;
- the sealed record has exactly packet digest and answer-key fields, with
  matching packet identity and committed answer-key hash;
- the packet declares the registered schema and contains exactly 64 cases;
- all 64 case identities are nonempty strings, mutually distinct;
- the sealed answer-key is a JSON object covering **exactly** the same
  64 case identities; and
- every answer is exactly one `A`, `B`, `C` or `D` string.

Malformed shapes now return sanitized `HOLD EXP053` instead of raw lookup or
container exceptions. The normal 40-article scientific **acceptance** minimum
still applies to the final 64-source case cohort; the check does not make
40/64 artificially successful and does not waive adjudication or signer
requirements. Existing digest and signature custody checks stay mandatory.

The tests use only invented case IDs, fabricated answer strings and mocked
boundaries. No scientific admission or model execution took place; no human
identity, credential custody, or independently signed review was asserted.

```
python3 -m pytest -q tests/test_exp053_sealed_case_scope.py
```

This exact code change needs independent qualified technical review before
any merge. Reviewer A/B enrollment, 40 genuinely co-reviewed/adjudicated
articles and separate witness/weight control remain blocked.
**NO_GO_RESEARCH_ONLY.**
