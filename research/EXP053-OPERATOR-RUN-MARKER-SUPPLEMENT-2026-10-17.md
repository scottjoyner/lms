# EXP-053 operator run-id reuse supplement — frozen 2026-10-17

This supplement adds an independently owned run-ID marker requirement to
the pre-frozen run-sequence protocol, which is otherwise unchanged.

An operator-signed authorization with a fresh sequence number is NOT sufficient
if it names an already accepted scorer execution ID. Therefore each accepted
authorization must cause the creation of exactly one private single-use marker
alongside the operator's pre-existing locked claim journal.

Marker filename: `<journal basename>.used-<runner_execution_id_sha256>`.
The runner ID SHA is an exactly lowercase 64-hex SHA256 fingerprint already
bound to the signed scorer execution receipt. Marker creation uses O_CREAT,
O_EXCL, O_NOFOLLOW with permissions 0600, under the separately approved
non-group/non-world-writable operator directory. Marker content is the
SHA256 of the complete accepted operator-signed authorization plus newline.

Marker file and its parent directory must be fsynced before persisting the
new last_sequence/last_claim_sha256 journal state. A conflict (the marker
already exists) must HOLD and must not advance the sequence. A crash after
marker creation but before journal commit conservatively blocks reuse
until independent operator audit: the code must NEVER delete a marker
automatically. Existing markers remain durable across later valid claims.

This protects against a re-signed authorization reusing the SAME scorer
execution ID, not just reuse of the same signed authorization file.
The state journal is still required to reject out-of-order/forked claims.

This is NOT resistant to rollback of the entire operator directory by a
privileged adversary. External human-owned append-only custody, independent
high-water/anti-rollback evidence, native model execution validation, distinct
signing keys, and zero implicit scientific authorization remain mandatory.
No actual EXP-053 model forward, key enrollment or production writes allowed.
