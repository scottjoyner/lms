# EXP-053 — offline human review signer input boundary finding

**2026-10-19, source-only synthetic negative tests. NO_GO_RESEARCH_ONLY.**
No actual reviewer credentials, licensed RACE passages, signatures, or models
were opened or dispatched by this work.

## Reproduced gap

In the old `exp053_sign_review.sign()` an adjudicator draft containing
`decisions=null`, `decisions={}`, or a list-valued `case_id` raised unsanitized
`TypeError`/`AttributeError` before the signer could reach the explicit review
validator. Independently, packet and draft JSON were loaded with unbounded,
path-following `read_text()` and default `json.loads()` which accepts duplicate
object keys by keeping only the last value. These behaviors are input-safety
and review-integrity problems, NOT evidence of a real compromised reviewer.

## Narrow remediation

`read_offline_json()` now opens each chosen packet/draft final path via one
`O_RDONLY|O_NOFOLLOW|O_NONBLOCK` descriptor, requires a bounded regular file
<=8,000,000 bytes (the existing offline HTML import size limit), checks
before/after device/inode/size/mtime/ctime, and closes the descriptor even
on injected read failures. It parses JSON with duplicate-key denial at all
nesting levels, strict UTF-8, non-finite-number denial (including 1e999),
and object-only payloads. It does not print filename, key bytes or content.

`sign()` now checks the exact unsigned draft field set, intended schema,
nonempty pseudonymous reviewer identifier, personal attestation, permissible
role and list-shaped decisions BEFORE reading the signing key. Adjudicator
subsets reject unhashable, duplicate, or empty case IDs with a sanitized
`HOLD EXP053`. Existing `decisions_for()` validates source-packet binding and
per-case decisions. The human-operated signature and 0600-from-creation
exclusive signed-output writer are unchanged.

Tests use only disposable synthetic JSON, an optional ephemeral test Ed25519
key, negative races, an invalid FIFO, a symlink to a fixture, duplicate keys,
large inputs, malformed adjudicator decisions, and one valid **synthetic**
signed review. No sample is admitted to EXP-053; no owner roster or source
manifest is changed, and no witness or production system is deployed.

## Limits and next independent action

The source-only input checker does not attest secure storage of the browser's
locally exported draft, independence of a human reviewer, absence of
same-user process tampering, or trustworthy ancestor directory components.
The human must store licensed packet/draft privately, independently control
an owner-enrolled signing key and review in a blinded environment. Until
independent technical and scientific review is complete, reviewer enrollment,
40 independently co-reviewed source-disjoint RACE items, separate independent
custodian/rollback authority and all inference remain on HOLD.

Review reproduction:

`python3 -m pytest -q tests/test_exp053_signer_input_boundary.py`
