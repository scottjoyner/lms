# EXP-053 isolated mutual-TLS witness fixture — frozen 2026-10-17

**Prospective TEST-ONLY contract**, frozen before implementation. The live
external witness adapter and all science authorization pins remain empty.

## Scope

Extend the existing synthetic, operator-authorized SQLite CAS with a temporary
loopback TLS 1.3 server on 127.0.0.1, ephemeral port zero, solely within
unit tests under `tests/fixtures/`. All credentials and SQLite state are
generated in disposable test directories. No real service, network exposure,
deployment, fleet routing, production integration or model inference.

The server and client use a fresh test certificate authority (CA) and distinct
CA-issued server/client leaf certificates. Require TLS 1.3, server hostname
verification (SAN IP 127.0.0.1), client certificate verification
(`CERT_REQUIRED`), and a separately configured exact SHA256 of the
authorized client leaf certificate on the SERVER. A different CA, missing
client certificate, or different client certificate from the SAME CA must
not advance the SQLite counter. The client validates the server CA, SAN
and certificate-chain validity. Reject an unrecognized server.

After successful TLS mutual authentication and server-side exact client
leaf pin, read ONE length-prefixed, source-free JSON CAS request with a
maximum 32 KiB body; require strict JSON (duplicate keys and
NaN/Infinity rejected), exact frozen authenticated-request fields and
bounded Ed25519 operator authorization payload BEFORE `BEGIN IMMEDIATE`
or any SQLite state transition. Return at most one signed committed
response using the existing response schema. Any exception closes the
connection without a fabricated success. A network timeout or failure
must not reset/repair durable witness high-water.

Every test server handles at most one connection, binds only to loopback,
has a finite accept/read timeout, and is joined/closed before test teardown.
No certificate/key bytes or private SQLite DB may be committed to Git.

## Synthetic required evidence

Test one legitimate TLS 1.3/mutual-authenticated, Ed25519-authorized CAS
and its signed response, persisted SQLite sequence; no client cert;
unauthorized same-CA client leaf; untrusted-CA client; wrong server hostname
or untrusted server CA; tampered CAS operator signature or malformed JSON;
oversized request; repeated claim with a fresh TLS connection. On every
denial, the SQLite high-water and number of committed claims must stay
unchanged.

Fixture isolation means the client/server/SQLite/key material can be
compromised together. The fixture does NOT establish independent off-host
physical custody, trusted production TLS roots, certificate revocation,
actual native inference, signed human reviews, or rollback-resistant
operator authorization. The real `research/exp053_external_witness.py`
default adapter MUST continue to deny; all real approval pins remain EMPTY.
No production model, NAS/Neo4j or fleet write is permitted.
