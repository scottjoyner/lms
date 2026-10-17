# EXP-053 separate-process loopback witness fixture — frozen 2026-10-17

**Only synthetic research integration tests. No production or externally
administered witness, human approval, inference, or model deployment.**

## Motivation and trust limitation

The prior one-request TLSv1.3 test server ran in a thread in the SAME Python
process as the client. A separate OS process can detect accidental dependence
on in-process Python objects, shared locks, and transient SQLite handles.
Process isolation on ONE host is NOT an independent trust domain.

## Fixture contract (tests/fixtures ONLY)

- Each test generates disposable X.509 CA, server and authorized client
  certificates, a private SQLite CAS DB, ephemeral Ed25519 witness and
  operator signing keys. Never upload generated credentials, database or
  licensed article content to public Git.
- The witness server starts in a fresh Python `spawn` child and opens its own
  SQLite connection(s). Its startup readiness pipe reports a loopback
  127.0.0.1 port zero allocation; no nonloopback binding, daemon, reusable
  service account or dynamically discoverable LAN endpoint.
- The process requires TLSv1.3 and `CERT_REQUIRED`, validates exact
  authorized client leaf-certificate SHA256 after ordinary chain checks,
  reads a <=32 KiB single length-prefixed strict JSON CAS request,
  independently verifies pinned test operator Ed25519 authorization and
  commits the SQLite highwater CAS before signing its reply. On a denial:
  close without any success reply or SQLite update.
- Client separately validates server CA, hostname/SAN, TLSv1.3 and exact
  committed response signature. Test processes have strict timeouts and
  must be joined/terminated on teardown.
- Restarting the witness process MUST retain SQLite state but NOT any Python
  singleton state. Duplicate claims and local journal/marker rollback
  MUST be rejected across witness process restarts.
- A deliberately killed test server AFTER durable SQLite COMMIT but BEFORE
  writing its TLS response MUST not erase the claim. The client experiences
  uncertainty/failure; replay of the same claim after restarting the
  witness fails. Never fabricate success or auto-reset the highwater.
- A server crashed BEFORE COMMIT must leave SQLite highwater unchanged
  (test fixture may deliberately crash at the precommit hook).
- A server without the right operator authorization, TLS certificate,
  client pin, or trusted server identity MUST NOT advance the database.
- Optional simultaneous child witness servers sharing the same disposable
  SQLite file demonstrate actual cross-process transactional CAS; exactly
  one may accept a competing sequence.

A final source-only test verifies that the test fixture is NOT imported by
`research/exp053_external_witness.py`. All production trust root pins remain
EMPTY and the production CAS adapter remains a hard explicit DENY.

**Important:** Tests may explicitly transfer a synthetic private signing
seed to the child. That seed is disposable and MUST NOT be used as a real
operator/witness key or interpreted as independent key custody.
The fixture does not prove remote mTLS revocation, hardware key custody,
off-host rollback resilience, service identity, or real model execution.
