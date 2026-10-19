# EXP-053 source-free receipt collection consistency handoff — 2026-10-18

**No scientific authority:** `NO_GO_RESEARCH_ONLY`. The physical candidate
SSH preflight and controller identity audit are not equivalent to independently
controlled custody. No human-administered witness, external high-water CAS,
durable published checkpoint, independently signed negative tests, or accepted
RACE human adjudication has been established by this continuation.

## What changed

The prospective, independently frozen contract
`EXP053-CUSTODY-CROSS-BUNDLE-REPLAY-SPEC-2026-10-18.md` (SHA256
`33774931fce461664335e3c3e6402c838266cf2d7ba543849f709f93d485d296`)
defined a bounded read-only **negative** cross-bundle audit before code was
implemented. `exp053_custody_cross_bundle_replay.py` now accepts source-free
custody claims and at most 12 signed-claim bundles through `--claims` and
`--bundles`. Each document is validated by the existing frozen custody,
prospective challenge plan and receipt-consistency validators; Ed25519
signatures are checked only against the observer key embedded in the bundle.

For any given plan, it rejects normalized duplicate bundles, repeated scenario
IDs and nonces across bundles, swapped observer keys, claimed checkpoint
forks, malformed signatures, mixed empty/nonempty baselines, stale/future
self-reported times and all upstream malformed inputs. It never stores a
trusted high-water, writes source data, creates an authority key, contacts
fleet nodes, or executes negative physical tests. A report contains no signing
keys, key fingerprints, evidence hashes, paths, receipts, signatures or
private text. Every CLI invocation returns exit code **2**.

## Lenovo reality / source-free baseline

The existing zero-receipt Lenovo fixture remains the only current sample.
`python3 research/exp053_custody_cross_bundle_replay.py --claims research/EXP053-LENOVO-CUSTODY-CONTROL-CLAIMS-2026-10-18.json --bundles research/EXP053-LENOVO-UNATTESTED-RECEIPTS-2026-10-18.json`
returns: 1 bundle, 0 unique signed scenario claims, 12 scenarios missing,
`signer_authority=UNTRUSTED_SELF_SUPPLIED_KEY`,
`durable_replay_protection=NOT_PROVEN`, and **scientific NO-GO**.

The synthetic adversarial test suite deliberately signs two source-free
bundles covering 12 disjoint challenge IDs. Even when all twelve signatures
are cryptographically correct and all outcomes claim denied access, the audit
returns `all_challenges_independently_attested=false`,
`independent_administration_approved=false`,
`witness_durability_approved=false`, `scientific_execution_authorized=false`.
This ensures testing progress cannot be mistaken for custodian approval.

## Critical limitations / independent next step

This is **NOT durable replay protection**. It compares only files the caller
submits now. An operator can silently omit a prior receipt, rewrite an entire
source-free history, forge a new self-signed observer key, make up a checkpoint
hash, reset the local clock or restore the old collection. The tool cannot
notice those changes without independent out-of-band, append-only high-water
checkpoint publication and a pinned independently administered trust root.
The same self-supplied observer key is not evidence that an individual was
independent. Neither claimed denial nor a valid signature proves the physical
challenge was performed. Hardening a local verifier cannot substitute for
separate custodianship.

**Next true gate:** an independent human custodian provisions and controls a
separate *disposable nonproduction* witness, with independent signer/key
custody, native durable high-water CAS, external publication, snapshot and
backup authority separation, negative privilege tests, verifier-pinned
checkpoint chain and a recovery/revocation plan. Independently reviewed RACE
labels >=40, model-byte receipts and scorer source custody are still separate
HOLD conditions. Do not run EXP-053 model inference or change any approved
key/adapter pins until those conditions receive their own approvals.

This work used only controller-side Git research files and synthetic
software-signing fixtures. **No remote host, production service, model
runtime, SSH configuration, key material, NAS/Neo4j, deployment or GitHub PR
was changed or promoted by this continuation.**
