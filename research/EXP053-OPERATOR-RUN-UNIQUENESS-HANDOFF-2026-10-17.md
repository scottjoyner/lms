# EXP-053 research-only operator run uniqueness handoff — 2026-10-17

## Finding closed in source-only software

EXP053 already requires a private, signer-attested score file; a matching
score-row attempt ledger; human source-adjudication; and pinned model,
tokenizer and prompt identities. None independently proves that the same
signed execution wasn't credited twice, even if all data and signatures are
valid. This weakness remained after retry-log validation.

We prospectively froze two immutable source-free contracts BEFORE writing
the verifier:

* `EXP053-OPERATOR-RUN-UNIQUENESS-SPEC-2026-10-17.md`
  SHA256 `8b3999c0febd6269d8a96f0ee31c719736c32aee95b769abc201f7beb51bada9`
* `EXP053-OPERATOR-RUN-MARKER-SUPPLEMENT-2026-10-17.md`
  SHA256 `fdccbe3b99fb11eba77a6f7f83c759a1b740e1a12c7e0cd07d4aeca29401049a`

New `research/exp053_operator_run.py` enforces an independently
operator-signed Ed25519 authorization bound to the previously verified
runner execution ID SHA256, complete runner attestation hash, exact human
packet and score archive. The operator signing key must be distinct
from the scorer key.

No caller supplies an operator root or journal path. These
**source-pinned roots remain intentionally unconfigured**:

* `APPROVED_OPERATOR_PUBLIC_KEY_SHA256=""`
* `APPROVED_OPERATOR_CLAIM_LEDGER_PATH=""`

Once independently approved, the operator-owned journal must be a
PRE-EXISTING, mode-0600 regular file outside the public Git repository and
outside scorer-controlled private evidence. The verifier NEVER initializes
the ledger for an unauthenticated caller. It takes exclusive `flock`,
checks file identity/owner/permissions and strictly serialized JSON,
requires exactly the next sequence with the previously accepted signed
authorization checksum, and writes/fsyncs the monotonic state only
after creating and fsyncing a distinct **O_EXCL one-use marker** named
for the execution-ID hash. Existing markers are never removed.

A duplicate authorization, a newly operator-signed authorization with
the same execution ID, a stale sequence, a forked prior checksum, or
two racing consumers for one run must fail HOLD. A crash after creating
a marker but before updating the ledger remains a deliberate, conservative
HOLD until independent manual recovery; code never resets high-water
state or deletes the marker.

`exp053_verified_analysis.py` now requires
`--private-operator-authorization` and invokes the operator claim
ONLY AFTER all human, weight, scorer, score-row, retry and statistical
validation and final private score/log hashes succeeded. The claim
is consumed BEFORE writing the source-free aggregate; failures
in output persistence cannot reuse the same claim. The only
additional public output is operator authorization SHA256 and sequence.

All adversarial tests use **ephemeral software-generated signing keys and
disposable isolated filesystem directories**. They are NOT actual human
approvals, native execution evidence, independent replication, RACE labels
or real model-score observations.

## Remaining SCIENTIFIC NO-GO

This protects one local operator journal against duplicate claims only
as long as its state and marker directory cannot be rolled back, copied
or forged. A privileged actor can restore an earlier snapshot and replay.
A key holder can falsely attest to a run. Real independent publication
requires an EXTERNAL trusted high-water/custody witness, physical
model-load + runtime process evidence, independent human enrollments,
operator-controlled durable key custody and source-safe negative controls.

Current research gate constants all remain empty:

- `APPROVED_REVIEWER_ROSTER_SHA256=""`
- `WEIGHT_REPORT_SHA=""`
- `APPROVED_RUNNER_PUBLIC_KEY_SHA256=""`
- `APPROVED_RUNNER_SOURCE_SHA256=""`
- `APPROVED_OPERATOR_PUBLIC_KEY_SHA256=""`
- `APPROVED_OPERATOR_CLAIM_LEDGER_PATH=""`

The 39-file checkpoint hash audit remains a valid preparation result
but NOT independent authorization for model forwards.
There are **0 independent human-reviewed articles and 0 new EXP053
model forward passes**. PR #32 remains draft/unmerged; no RACE
licensed source, private keys, reviewer identities, native logs or
per-case model scores belong in public GitHub. No NAS/Neo4j writes,
training, serving, deployment or scientific promotion.
