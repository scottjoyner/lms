# EXP-053 mutual TLS external-witness prototype — 2026-10-17

## Scope

**Synthetic, nondeployable loopback-only TLS witness test fixture**, extending
the previously verified SQLite CAS + Ed25519 operator authorization.

Before implementation the exact prospective test contract was frozen:
`research/EXP053-LOOPBACK-MTLS-WITNESS-SPEC-2026-10-17.md`,
SHA256 `67de58c79ec9b99de588a022a152b21fcdcd00461fbd9111d88e4fc50e9fb87d`.
The original public source-only high-water protocol remains unchanged.

Added `tests/fixtures/exp053_mtls_loopback_fixture.py` and
`tests/test_exp053_mtls_loopback.py`. Each test generates its own
ephemeral ECDSA CA, CA-signed TLS 1.3 server and client certificates
with correct EKUs/SANs, a distinct temporary operator Ed25519 key,
a distinct temporary witness signing key, and an isolated SQLite file.
The one-connection server binds **only 127.0.0.1**, port 0, with short
socket timeouts, bounded 32 KiB length-prefixed source-free JSON, and
returns a signed CAS success only after TLS client authentication AND the
independently pinned test client certificate SHA256 AND the exact
operator-signed Ed25519 authorization check. No port or cert exists after
the test returns; no keys or database contents belong in public Git.

The client verifies server CA and SAN/hostname and requires TLS 1.3.
The server uses `CERT_REQUIRED` and only permits a specific pinned
client certificate (not any otherwise valid same-CA leaf). A different
CA, missing cert, different authorized-CA client identity, invalid
server trust root or wrong hostname must be denied before any state
transition.

The fixture exercises the **real source client response validator**
`research/exp053_external_witness.py` only with test-time injected,
short-lived roots and a callback which opens the disposable TLS socket.
It never replaces the default production source adapter.

## Testing and interpretation

Test-only positive control: an authenticated TLS 1.3 request with exact
signed operator authorization causes one persisted SQLite CAS advance
and a valid source-verifiable signed witness response.

Negative controls: missing client cert, wrong CA, wrong leaf under
the valid CA, wrong server hostname/CA, corrupted operator signature,
ambiguous authorization JSON, oversized frame, replay over a second
authenticated TLS session, and generated cert private-key permissions.

These controls establish **local software protocol behavior only**,
not real external host separation or independent key custody. The
test server and SQLite file are both created by the same research process
on x1-370, so an adversary who controls that process or its filesystem
could forge or roll back everything. Real remote authorization requires
independently provisioned and reviewed mTLS roots, certificate expiry/
revocation, key isolation, reliable off-host persistence, native-run
physical witnesses, and signed human-reviewed scientific admission.

All actual `APPROVED_*` roots remain EMPTY; production
`_external_cas` remains an explicit DENY with no sockets or adapters.
There were **zero human-approved EXP053 cases and zero new EXP053 native
model forwards**. No model training, deployment, RACE raw passage upload,
NAS/Neo4j writes, or public performance claim is permitted.
PR #32 remains draft and unmerged.
