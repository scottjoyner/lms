# EXP-053 prospective custody-boundary negative audit spec — 2026-10-18

**Frozen before implementing the controller-only parser.** This protocol is a
research-only *negative* audit for a future off-host witness, not an approval
path, a service, or proof of administrative independence.

Input is source-free JSON bounded to 16 KiB, UTF-8, exact schema
`exp053-custody-boundary-claims-v1`, exact top-level keys `schema`,
`candidate_alias`, `claims`. A candidate alias is an explicit safe SSH alias
matching `[A-Za-z0-9][A-Za-z0-9_.-]{0,62}`; no IPs, domains, users, key
names, paths, actual principal identities, RACE passages or private data.

Each claim has EXACT keys: `domain`, `principal`, `access`, `evidence_sha256`,
`evidence_origin`. Domains are restricted to:
`witness_remote_login`, `witness_root_administration`,
`witness_signing_key_custody`, `witness_highwater_mutation`,
`witness_snapshot_restore`, `witness_backup_restore`,
`witness_trust_anchor_rotation`, `witness_checkpoint_publication`.
Principals: `scorer_operator`, `fleet_automation`, `witness_custodian`.
Access: `CAN`, `CANNOT`, `UNKNOWN`. Evidence origin:
`operator_observation`, `custodian_claim`, `synthetic_fixture`, or `none`.
`evidence_sha256` must be a lowercase SHA256 or JSON null; if null, the
origin must be `none` and access MUST be `UNKNOWN`. If non-null, origin
must not be `none`. This is only a **claimed digest reference**, never
cryptographically verified by this parser. A distinct human or organization
must independently check the private original evidence, signature, custody,
replay resistance, administrative rights and recovery procedures.

Reject duplicate JSON keys, duplicate `(domain,principal)` rows, unknown
keys/values, booleans in numeric fields, nonfinite values, oversized input,
null input, untrusted characters and missing/malformed digests. Require 1–24
claims; omitted role/domain pairs are UNKNOWN, never implicitly CANNOT.

Report only source-free fields, stable classifications and counts. `CAN` for
`scorer_operator` or `fleet_automation` is a *claimed shared-control risk*;
`CANNOT` with digest is still an *unverified negative claim*. An empty report,
all-CANNOT report, solely custodian claims, and a report with no detected
shared controls must all remain `NO_GO_RESEARCH_ONLY`. The CLI MUST return
exit code 2 for any well-formed report (a nonadmitting negative screen).
On malformed input it MUST refuse with no raw input or private path leakage.

Every report MUST include unconditional:
- `independent_administration_approved=false`
- `witness_durability_approved=false`
- `scientific_execution_authorized=false`
- `decision=NO_GO_RESEARCH_ONLY`

Never interpret presence/absence of these claims as durable external CAS,
protected key storage, independent human signoff, or authorization to infer.
Only an independently governed external attestation+native negative testing
can later establish those facts through separately reviewed code; this tool
has NO route to change approval pins. No live SSH probing, production writes,
model loading/forwarding, real key generation, NAS/Neo4j access, remote
installation, release, or PR merge is authorized by this workstream.
