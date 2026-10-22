# EXP-053 real fleet host qualification — read-only evidence

**Observation:** October 17, 2026, approximately 18:12 UTC.
**Scope:** read-only known-host SSH probes from x1-370 via Fleet Commander.
**Disposition:** all three are *physical candidates*, **NONE is an
independently approved witness**. No service, private signing key, new
network listener, directory, package, container, database or model run
was created on any remote host.

The prospective protocol was frozen before implementing
`research/exp053_witness_host_preflight.py`:
`EXP053-OFFHOST-READONLY-PREFLIGHT-SPEC-2026-10-17.md`
SHA256 `523cbece51e6d966e868e859e728946fe7c15990e83f428143d1a08fb5dceeb5`.

## Live physical candidate evidence

| Existing fleet alias | Python | SQLite | Crypto/openssl | Root filesystem | Approx free | TPM device | SSH round trip | Credential/custody verdict |
|---|---|---|---|---|---:|---|---:|---|
| lenovo | 3.12.3 | 3.45.1 | Available | ext4 | 40.4 GiB | Not exposed | 317 ms | NO-GO |
| destroyer | 3.12.3 | 3.45.1 | Available | ext4 | 74.2 GiB | Exposed | 283 ms | NO-GO |
| xwing | 3.12.3 | 3.45.1 | Available | ext4 | 81.2 GiB | Exposed | 249 ms | NO-GO |

Numbers are momentary, warm-cache read-only observations; not durability,
performance, TPM endorsement, secure-clock attestation or future
availability. Clock-offset observational error bounds were under 310 ms;
this only establishes coarse clock agreement, not secure timestamping.

The implemented preflight uses SSH `StrictHostKeyChecking=yes`,
`BatchMode=yes`, explicit aliases and bounded execution. It includes a
fresh challenge echoed in each read-only remote result, rejects
ambiguous/extra JSON fields, malformed values, untrusted/failed SSH,
missing cryptography/OpenSSL, Python <3.10, SQLite <3.35,
<2 GiB free, and excessive clock mismatch. Its entire result schema
explicitly includes `scientific_execution_authorized=false` and
`decision=NO_GO_RESEARCH_ONLY`. Tests are source-free and do not contact
real hosts.

## Recommendation and authority boundary

**Lenovo** is a plausible first host for a later *disposable, isolated*
off-host witness smoke test because it is reachable and is less tied
to current GPU workloads. Its absence of a visible TPM means a
software-held signing key would need an independently approved custody
design. **Destroyer** has a visible TPM device and more capacity, but
it is relatively occupied and has its own active fleet proxy role.
**Xwing** has a visible TPM device but carries other research and
inference workloads. TPM device presence alone does not prove usable
sealed signing or a meaningful security boundary.

Critically, **all three are reachable using the same existing fleet
administrative access path as x1-370**. Distinct machine hardware does
not establish independent operator administration. If those credentials
can administer, snapshot or reset the witness and scorer, the required
independent high-water and rollback-resistance guarantee is still absent.

Before physical research authority is granted, a responsible human
operator must independently designate the trust domain, attest
administrative isolation, establish separate roots and client/server
certificate custody, verify remote key isolation and hardware
security, approve a read-only-to-disposable test plan, and demonstrate
that the scorer credentials CANNOT alter/read/rollback the witness
state. A separate append-only high-water audit and native model-run
receipt review are also necessary. Genuine independent review of
>=40 original article cases and the checkpoint/scorer research
admission remain separate gating requirements.

**All existing real authority pins remain EMPTY**; no independent
human-approved cases, EXP053 model forwards, scientific performance
claims, training, production deployment, NAS/Neo4j writes or PR merge.
This qualification is a go/no-go research artifact, **not a permission
to deploy an external witness**.
