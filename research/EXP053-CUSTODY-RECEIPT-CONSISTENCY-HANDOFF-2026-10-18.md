# EXP-053 — future custody receipt consistency gate, October 18, 2026

**HOLD / NO_GO_RESEARCH_ONLY.** No human custodian has supplied authenticated
native negative-test results. No external witness or durable checkpoint has
been established. No model-forward execution is authorized.

## What was added

- Frozen source-free protocol `EXP053-CUSTODY-RECEIPT-CONSISTENCY-SPEC-2026-10-18.md`
  (SHA256 `73fa530a2704a857b449931150d48e63b7b15f3d07be0ca0055f2ba23268e36a`).
- Offline `exp053_custody_receipt_consistency.py`: strictly validate a bounded
  future receipt bundle against a deterministically generated 12-case plan;
  verify Ed25519 signatures over domain-separated, exact bound bytes;
  screen self-asserted timestamps against **only the local clock**; refuse
  duplicate scenario IDs/nonces, forged signatures, mismatched plan/alias,
  malformed input, or protocol drift. An untrusted signer public key is read
  only as source-free input and is never installed into approved authority pins.
- Zero-receipt Lenovo fixture
  `EXP053-LENOVO-UNATTESTED-RECEIPTS-2026-10-18.json` contains no keys,
  signatures, evidence, checkpoints or secret context.
- `tests/test_exp053_custody_receipt_consistency.py` tests in-memory
  disposable software signing keys and synthetic outputs exclusively. No
  test generates or installs a fleet credential or witness key.

## Current live-fleet implications — NONE

Controller-local Lenovo source-free receipt audit: 0/12 receipts, all 12
scenario IDs missing, `all_challenges_independently_attested=false`.
A complete 12/12 freshly signed synthetic bundle also returns **NO-GO**, even
when all twelve receipts claim DENIED. A signature only proves possession of
the **self-supplied key**; the observer can be the same operator, the receipt
can lie, the claimed evidence hash may not identify a real native test, the
checkpoint digest may be an invented value, and the local clock can be reset.
There is no persistent replay/high-water journal in this tool and no external
checkpoint lookup: cross-bundle reuse, key revocation, rollback and custody
cannot be proven by these checks. Any actual future authoritative gate MUST
be independently controlled and independently verified by separate reviewed
code and an external durable, monotonic checkpoint.

## Reproduction without inference or network access

```
python3 research/exp053_custody_receipt_consistency.py \\
  --claims research/EXP053-LENOVO-CUSTODY-CONTROL-CLAIMS-2026-10-18.json \\
  --receipts research/EXP053-LENOVO-UNATTESTED-RECEIPTS-2026-10-18.json
```

Expected exit **2**, `cryptographically_consistent_untrusted_receipt_count=0`,
`scientific_execution_authorized=false` and
`independent_administration_approved=false`.

The previously documented shared controller SSH credential material remains a
separation concern; it is not proof of root access and does not qualify Lenovo
as an independent witness. A separately accountable custodian must provision
an isolated nonproduction environment and authorize/perform the 12 negative
tests; they must maintain their own signing-key administration, signed
high-water publication, immutable evidence and recovery path. Even after those
external steps, EXP-053's >=40 independently adjudicated article-disjoint
items, approved exact runner/model-byte custody and separate scientific gates
remain blocked. All existing production trust pins remain intentionally empty.

**Never deploy, publish private evidence, change remote hosts/services,
write NAS or Neo4j, merge/promote a draft, or execute models on the strength
of this verifier.**
