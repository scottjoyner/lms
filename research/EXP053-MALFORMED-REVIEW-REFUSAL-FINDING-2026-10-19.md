# EXP-053 — signed-review malformed-input refusal finding

**October 19, 2026. NO_GO_RESEARCH_ONLY.** Source-only negative tests. No
independent human review, source transfer, key-enrollment or neural inference.

## Confirmed and reproduced behavior

Prior `exp053_race_review.decisions_for()` attempted dictionary/set membership
on each untrusted `case_id`. A synthetic review containing a list or JSON
object as its case ID caused an uncaught Python `TypeError` instead of a
sanitized `HOLD EXP053`. A synthetic trust mapping with list/dict public-key
values likewise threw `TypeError` at `set(trust.values())` before reaching
signature validation. These were **malformed input handling defects**, not
verified signature bypasses or evidence of fraudulent human participation.

## Narrow remediation

The validator now refuses missing, duplicated, empty, or non-string packet and
decision case IDs before hashing them as keys, confirms each source case has a
string article, and checks that adjudicator subsets contain unique string
case IDs. The trust-store validator requires nonempty string reviewer IDs
and public-key strings before checking key reuse. The existing Ed25519
signature verification, exact source-packet SHA, A/B enrollment roles,
independent third adjudicator requirements and minimum approved articles
remain unchanged. All invalid cases fail with a deterministic
`ValueError("HOLD EXP053: ...")` rather than a raw unhashable-object error.

Tests use only invented synthetic one-line articles, fake reviewer decisions
and invalid trust objects. No real human key or RACE material is opened.
Negative tests also prove the offline `exp053_sign_review.sign()` helper will
not reach the private-key accessor for malformed draft case IDs.

## Limits and next action

This repair does not prove review authorship, source provenance, independence,
license permission, safe custody of any signing root, or admissibility of a
single article. An actual accountable code reviewer must separately examine
this exact diff. A study owner and third-party custodian must separately
establish human key ownership and external trust before anything can proceed.
The public PR #32 remains draft, no independent review is recorded, the owner
reviewer-roster and model-weight pins remain unset, and EXP-053 inference is
still HOLD. No publication, merge or deployment is authorized.

Reproduction:

`python3 -m pytest -q tests/test_exp053_malformed_review_refusal.py`
