# EXP-053 — off-host independent custody negative-challenge handoff

**Research-only NO-GO; no model inference, physical negative tests or witness
operation were started.** This continuation provides a prospective checklist,
not additional affirmative approval evidence.

## New source-free, non-executable plan

`exp053_custody_negative_challenges.py` consumes the already reviewed
`exp053-custody-boundary-claims-v1` manifest and emits fixed
`exp053-custody-negative-challenge-plan-v1` JSON to stdout. It fails closed on
unknown source shape, overlarge or special-file input, symlinked final paths,
prohibited private fields, or frozen-source protocol drift. Every valid run
returns exit status **2**, never zero. Its emitted `input_sha256` is only a
content correlation handle, not proof of custody or an authorized signature.

The code never calls SSH, an agent, remote storage, key generation, the model
runtime, or a third-party witness. There is no mechanism to mark a test as
run or approved. Source code and the frozen spec have no production adapter,
credential selector, network target, or test-execution commands.

### Controller-side Lenovo starting point

Input: `EXP053-LENOVO-CUSTODY-CONTROL-CLAIMS-2026-10-18.json`.

| Category | Observed source-free plan |
| --- | ---: |
| Defined negative test scenarios | 12 |
| Scenarios actually executed | **0** |
| Scenario evidence independently attested | **0** |
| Scenario roles with missing claims | 8 |
| Scenario roles with unknown claims | 3 |
| Scenario roles with claimed shared access | 1 |
| Independently administered witness | **Not established** |

Shared controller login is a **risk indicator for a future witness**, not
proof of login to a deployed witness or of sudo/root authority. All other
privileged controls, signer custody, backup/restore access, independent
checkpoint publication and monotonic external CAS durability remain unknown.

## Future custodian acceptance path — no self-certification

An independent custodian must own the *separate* nonproduction environment,
witness administration, evidence and signing keys, plus an external high-water
publication channel. After written human approval, the custodian can design
controlled tests for each scenario in
`EXP053-INDEPENDENT-CUSTODY-NEGATIVE-CHALLENGES-SPEC-2026-10-18.md`.
Tested scorer and fleet principals must be isolated from witness root,
signer, immutable high-water and snapshot/backup recovery authority.

Capture independent negative evidence for both authorized and unauthorized
states, unique run identifiers, signer and environment provenance, replay /
rollback / crash-recovery rejection, and separately published high-water
sequence checkpoints. Do not treat the simple SHA256 references from the
controller as authenticated observations. No fixture, local key, Git commit or
same-operator attestation can certify the independence requirement.

Once this independent evidence exists, it needs an entirely separate
operator-reviewed, cryptographically anchored admission design. The prospective
plan here must remain a permanent NO-GO instrument, not itself gain an
approval mode. Human RACE source adjudication >=40 items and distinct
scorer/model-byte custody remain separate scientific blockers.

**No branch merge, push, service change, deployment, remote file write,
key transfer, actual witness challenge, model forwards, or NAS/Neo4j write
was conducted by this workstream.**
