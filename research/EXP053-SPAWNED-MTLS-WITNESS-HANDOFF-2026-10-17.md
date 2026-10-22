# EXP-053 spawned-process synthetic witness — October 17, 2026

## Scope and source isolation

Implemented **only** a disposable Python `spawn`-process TLSv1.3
witness fixture in `tests/fixtures/exp053_spawn_tls_fixture.py` with
corresponding `tests/test_exp053_spawn_tls.py`. The protocol was
prospectively frozen before code:
`EXP053-PROCESS-ISOLATED-MTLS-SPEC-2026-10-17.md`, SHA256
`eff7e93d3aa4744c4027c2171f93d301bacb172c5daa4cb3316839d09c325c3b`.

The prior loopback TLS test ran a thread in the client's Python process.
The new fixture starts a distinct, short-lived **OS process** using
`multiprocessing.get_context("spawn")`. The child constructs its own
SQLite CAS fixture and transaction connection, binds a single socket only
to `127.0.0.1:0`, and announces its ephemeral port and distinct PID through
a private startup pipe. Both sides require TLSv1.3, the server requires
a CA-valid client certificate and an exact approved leaf-certificate SHA256,
and the independently pinned synthetic operator Ed25519 authorization is
verified before SQLite CAS. All certificates, signer seeds and DB files
exist exclusively in a disposable directory; no secrets go to GitHub.

## Evidence

Fully source-free synthetic scenarios include:

- OS PID separation, one valid signed CAS and SQLite persistence after child
  exit; independent client-side production-format *response verification*
  using test-injected pins.
- Server **process restart** rejects the previously accepted exact claim
  based on SQLite rather than a Python singleton.
- Interrupted transaction *before COMMIT* leaves high-water at genesis,
  allowing the operator's same signed request to succeed after restart.
- Deliberate child-process **exit immediately AFTER SQLite COMMIT but BEFORE
  response**: client sees a failed/ambiguous operation, process exits with
  test-only code 73, and new child refuses the duplicate. No forged success.
- Missing authorized client certificate, different valid same-CA certificate,
  invalid server hostname or CA, and corrupted Ed25519 operator signature
  all refuse with zero database advancement.
- Two independent child witness processes, each with its own TLS socket
  and DB connection, race on the same fresh CAS sequence. Exactly ONE
  commits, the other refuses; SQLite's monotonic sequence and unique
  claim table remain correct.

Server and client have bounded socket timeouts; test processes are joined,
with explicit termination on a hung test-only child. No persistent daemon,
production module import, external network listener or API deployment.

## Scientifically important limitations

The witness child receives a **synthetic ephemeral witness signing-key
seed from the parent test**, so this tests process isolation, *NOT*
independent key custody. Both processes are on x1-370 and both use a
locally writable disposable SQLite file. Thus privileged rollback of
that database remains possible. No independent machine, operator, durable
off-host append-only high-water or human-reviewed article approvals exist.
The production `research/exp053_external_witness.py` default callback
continues to DENY and all actual `APPROVED_*` roots remain EMPTY.

No new EXP-053 native model forward, licensed RACE corpus disclosure,
training, NAS/Neo4j write, model deployment or scientific performance
claim was made. Keep PR #32 DRAFT and unmerged until genuine independent
human and physical evidence are reviewed.
