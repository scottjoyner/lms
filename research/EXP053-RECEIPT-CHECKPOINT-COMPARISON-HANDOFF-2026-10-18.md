# EXP-053 prior receipt checkpoint comparison — October 18, 2026

**NO_GO_RESEARCH_ONLY.** No independently controlled custodian, published
checkpoint, committed external CAS, or signed native negative tests exist.
This continuation adds a read-only negative reference comparator, NOT durable
rollback resistance, external witnessing or scientific authorization.

## Code and frozen protocol

- `EXP053-RECEIPT-CHECKPOINT-COMPARISON-SPEC-2026-10-18.md` was frozen
  before implementation, with SHA256
  `aa7d4cd22df8a8b9a094c0172789398560672e38271b5fb60e9e11918d22f914`.
- `exp053_receipt_checkpoint_comparison.py` accepts one previously captured,
  source-free reference and requires an exact SHA256 literal separately
  supplied on the command line. It rejects altered files, malformed or
  noncanonical reference encodings, changed plan/alias, duplicated/reordered
  IDs, omissions or substitutions of earlier entire signed receipt rows,
  invalid source signatures, stale current self-reported dates, and collection
  conflicts. All input files are local, bounded, opened via O_NOFOLLOW, and
  read-only; no key bytes or private evidence are extracted.
- `tests/test_exp053_receipt_checkpoint_comparison.py` signs ONLY synthetic
  in-memory Ed25519 cases, exercises all twelve cases, and models a rollback
  of the scorer-supplied checkpoint *and* its hash in the same transaction.
  That attack remains UNDETECTABLE here: it is an explicit negative result,
  not proof of operational recovery or independent anti-rollback trust.
- `EXP053-LENOVO-UNTRUSTED-EMPTY-REFERENCE-2026-10-18.json` is an untrusted,
  empty prospective checkpoint fixture, with zero signed receipts. It must
  not be used to bootstrap an actual witness or to assert prior publication.

## Safe local reproduction

The local Lenovo sample still has 0 of 12 independently attested receipts.

```
python3 research/exp053_receipt_checkpoint_comparison.py \\
 --claims research/EXP053-LENOVO-CUSTODY-CONTROL-CLAIMS-2026-10-18.json \\
 --bundles research/EXP053-LENOVO-UNATTESTED-RECEIPTS-2026-10-18.json \\
 --reference research/EXP053-LENOVO-UNTRUSTED-EMPTY-REFERENCE-2026-10-18.json \\
 --expected-reference-sha256 8c9573fae19ca9f0a268f8393e03c2879397eccdf6870ca27f800898617ab4e9
```

Expected exit code **2** and JSON with `historical_receipts_consistent=true`,
`prior_receipt_count=0`, `current_untrusted_receipt_count=0`,
`reference_custody=CALLER_SUPPLIED_UNTRUSTED`,
`durable_replay_protection=NOT_PROVEN`, and all authorization flags false.
The SHA256 literal is a fixture checksum, NOT an independent trust root.

## Remaining physical and independent verification

The controller itself can rewrite both a local reference and the CLI digest.
A compromised operator can also omit the independent publisher entirely,
replay an older pair of files, change the clock or forge a new self-signed
observer key. This comparator has zero persistence and no external readback;
it cannot settle any of these attacks. The next true validation must involve
an independently administered, immutable publication channel and a verified
out-of-band pinned observer root, with independently executed nonproduction
rollback and privileged recovery tests. Human source adjudication of at least
40 article-disjoint RACE items and independently approved model/scorer custody
remain separate pre-inference blockers.

No SSH sessions, network calls, production service changes, witness setup,
new real keys, model inference, NAS/Neo4j writes, merge, or release are
performed by the comparator or its tests. It must remain permanently NO-GO.
