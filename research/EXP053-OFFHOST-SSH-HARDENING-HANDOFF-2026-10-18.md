# EXP-053 off-host SSH transport hardening — October 18, 2026

**Research-only. Operational candidate checks passed; scientific authority remains NO-GO.**

The October 17 prospective read-only protocol and its frozen SHA256 were not
modified. This change enforces a fresh SSH handshake (`ControlMaster=no`,
`ControlPath=none`), prevents automatic trust-store updates
(`UpdateHostKeys=no`), and prevents SSH-config local-command execution
(`PermitLocalCommand=no`) while retaining strict pinned-key checking,
BatchMode, disabled agent/port forwarding, explicit host aliases and bounded
timeouts. These flags protect the limits of this *preflight* only. They do NOT
establish independently administered witness or independent SSH key custody.

## October 18 live read-only requalification

Source: x1-370 through existing verified SSH aliases. One read-only Python
diagnostic ran on each remote candidate; no software, database, credentials,
model process, keys, witness socket or services were installed or changed.
Observed at 2026-10-18T15:32:55Z.

| Alias | Operational status | ext4 free GiB | Python | SQLite | Remote TPM device exposed | RTT ms | Clock-error bound ms |
|---|---|---:|---|---|---|---:|---:|
| lenovo | READ_ONLY_CANDIDATE | 40.7 | 3.12.3 | 3.45.1 | no | 302.0 | 290.8 |
| destroyer | READ_ONLY_CANDIDATE | 74.0 | 3.12.3 | 3.45.1 | yes | 312.4 | 233.2 |
| xwing | READ_ONLY_CANDIDATE | 81.1 | 3.12.3 | 3.45.1 | yes | 261.1 | 254.2 |

Each live report returned `NO_GO_RESEARCH_ONLY` with all independent
administration, custody, durability, and scientific authorization flags false.
Observed clock bounds are not secure-time attestation. TPM device exposure is
not evidence of sealed-key protection or independent administrative control.

## Verification and known environment limits

- Focused preflight unit tests: 12 passing, 19 subtests passing.
- All `tests/test_exp053_*.py`: 223 passing, 2 skipped, 68 subtests passing.
- Broad repository suite excluding legacy `test_k2*`: 454 passing,
  2 skipped, 68 subtests passing.
- Ruff clean for the two changed Python files, and `git diff --check` clean.
- Repository-wide research lint has four pre-existing `E401` import-formatting
  findings in other EXP-053 files, not altered here.
- Unrestricted `pytest tests` collection has 23 errors from missing legacy K2
  dependencies (`numpy`, `torch`, `transformers`) and absent K2 sources in this
  checkout. This is **not** a green full-repository test claim.

## Independence acceptance evidence still required (external to this change)

1. A separately accountable human custodian reviews and approves the witness
   physical/logical administration, signing-key storage, trust anchors, and
   loss/recovery plan; the scorer/operator cannot share unilateral rollback
   or destructive administration power over the witness.
2. Two-principal negative authorization checks demonstrate scorer credentials
   cannot read the witness private key, rewrite its high-water state, restore
   its disk from a stale image, or impersonate the witness; review snapshots,
   backup restoration, privileged interfaces and host compromise assumptions.
3. Independently published durable CAS high-water records and a native audit
   verify monotonicity and reject replay/fork/rollback across reboot/recovery.
   Self-signed claims on a machine administered by the scorer do not suffice.
4. Prospective corpus fixture hashes and a separately signed blind human
   adjudication by independent reviewers for >=40 article-disjoint cases
   precede ANY scoring. Retain the separate checkpoint and scorer signoff gates.
5. Only after external custody proof and operator review should a dedicated,
   disposable, nonproduction witness trial be considered. Do not reclassify
   Lenovo as independent merely because the SSH probe succeeded.

**All real EXP-053 witness approval pins remain empty.** No model forwards,
remote witness deployment, production routing/admission, Neo4j/NAS writes,
training, promotion or merge authorized by this handoff.
