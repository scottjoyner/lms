# EXP-053 per-cell attempt ledger handoff — 2026-10-17

**Research-only. No native EXP-053 model execution or genuine human approval.**

## Finding and change

Previously, future signed execution evidence bound `attempt_log_sha256`
to a private file, but replay did not validate its internal structure or
whether it described the same rows as the score archive. An arbitrary,
empty or self-inconsistent JSON retry log was therefore sufficient to
satisfy this single receipt field if an authorized signer signed it.

The prospective, source-free format has been frozen in
`research/EXP053-ATTEMPT-LEDGER-SPEC-2026-10-17.md`, SHA256
`5d9c94cf7efb094f1fcea5ef422d51907c64bad2c2438a24ddf619f8a1a1bba0`,
**before any native experiment or human review admission**.

New `research/exp053_attempt_ledger.py` requires one unique record for each
scored `(model, case_id, source arm, rotation)` tuple. Each record must bind
exactly the canonical SHA256 of its corresponding source-free score row.
Every cell must have 1–4 contiguous ordered attempts with either a single
final success or preceding explicitly classified failures. It rejects extra,
omitted or duplicate cells; early successes; gaps or booleans masquerading as
sequence numbers; unknown failure codes; extra private fields; mismatched
packet/run identity; and malformed, duplicated-key or non-finite JSON.
The private attempt file must be no larger than 8 MiB.

`exp053_verified_analysis.py` now calls this independent ledger verifier
**after validating every score row's source/label/weight receipt and before
statistical analysis**. It opens and hashes the same private file descriptor
against the previously signed execution SHA, and later rechecks both score and
attempt-log bytes before persisting any source-free public aggregate.
Only global counts of logged cells, failures and attempts are added to
the future aggregate; no raw attempts, case IDs or licensed source are sent
to public GitHub.

The 18 new software tests use entirely fabricated local score rows and retry
entries. They do NOT imply a valid signed scorer, real article adjudication,
physical runtime evidence, native GPU/CPU inference or independent science.

## Scientific and operator limitation

Even a fully matching ledger could be forged by someone controlling the
runner signing key. Completeness is relative to the *claimed* score grid;
it cannot prove all real attempts were logged. Ed25519 signatures and
source-free SHA256 cannot independently prove a native model ran or that
a reused execution ID represents a fresh replication. Operator-controlled
run uniqueness, trace custody, model process/weight residency and reviewer
identity remain separate scientific requirements.

The independent authorization pins for the human reviewer roster, scientific
checkpoint approval, scorer public key and scorer implementation remain
**intentionally empty**. No scorer implementation or EXP-053 neural forward
was run, no RACE source/answer text was published, and PR #32 remains DRAFT
and unmerged. Do not deploy, train or publish model-performance claims.
