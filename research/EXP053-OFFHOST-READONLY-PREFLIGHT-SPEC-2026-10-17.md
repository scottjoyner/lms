# EXP-053 physical off-host witness qualification — prospective read-only spec

**Frozen before implementing the qualification script on October 17, 2026.**

Purpose: identify potential **physical** host candidates via already configured,
host-key-verified, noninteractive SSH without deploying or creating ANY witness,
keys, socket, service, SQLite database, temp file, reservation or model run.

The script accepts explicit host aliases only, never performs dynamic host
scans, and runs an allowlisted read-only Python diagnostic over SSH
(`BatchMode=yes`, `StrictHostKeyChecking=yes`, bounded connection/
command timeouts). It does not send any credentials, repo files or RACE
content. No `sudo`, `ssh-keyscan`, remote writes, package installation
or service changes.

Collect from each host: Python and SQLite versions, availability of
`cryptography` and OpenSSL, exposed TPM device presence (not endorsement
of TPM suitability), root filesystem type and free space, host clock,
and randomized challenge echoed back to defeat cached/stale SSH command
responses. The source-free JSON report includes only the caller-given
nonidentifying aliases (not IPs, hostnames, machine IDs, users, addresses,
tokens or secrets), checks, RTT/clock-error *bounds*, and operational
preflight status. Reject malformed/extra JSON keys, missing data, wrong
challenge, wrong types, SSH host-key verification failure and unexpected
clock/skew evidence as explicit FAIL-CLOSED qualification outcomes.

Treat physical connectivity/preflight success ONLY as evidence of a
candidate's toolchain. Never infer independent administrative ownership,
TPM-backed private-key custody, external anti-rollback, network
isolation, credential separation, durable CAS, or operator approval.

Every report MUST state unconditionally:
- `independent_administration_approved` false
- `operator_key_custody_approved` false
- `witness_durability_approved` false
- `scientific_execution_authorized` false
- `decision` `NO_GO_RESEARCH_ONLY`

All response time/clock bounds are observational, not secure time
attestation. A private remote SSH user can spoof any machine-reported
capability. Script only validates the transport and format, not hardware.
If any candidate is unavailable, preserve separate denial evidence;
never silently drop failed hosts.

The host labels are user-selected. Report filenames may be committed
only after operator review confirms there are no network identifiers or
private values. The script and tests are public-safe, with entirely
synthetic remote stdout and subprocess mocks. No real witness
credentials, model forwards, reviewer admissions, NAS/Neo4j changes,
training, Git merge or production deployment are authorized.
