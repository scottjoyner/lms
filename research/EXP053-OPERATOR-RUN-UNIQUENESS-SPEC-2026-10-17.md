# EXP-053 prospective independent operator run-uniqueness protocol — 2026-10-17

**Frozen before implementation, real reviewer approval, any approved scorer, or new EXP053 model forward.**

## Problem

A valid scorer Ed25519 execution attestation and complete source-free attempt
ledger can still be replayed and counted repeatedly. The scorer's private key
does not establish a distinct physical run. A separate operator-issued
authorization and a persistent, operator-owned single-use claim are required.

## Approval roots — ALL UNCONFIGURED

* `APPROVED_OPERATOR_PUBLIC_KEY_SHA256`: independently verified operator
  Ed25519 public-key fingerprint; must differ from the approved scorer key.
* `APPROVED_OPERATOR_CLAIM_LEDGER_PATH`: separately approved absolute path
  to an existing PRIVATE regular state file outside the source repository
  and outside the scorer's private evidence directory. Neither tests nor
  a caller's CLI argument can configure either root in real operation.

The operator must create, custody and independently audit the ledger. The
scorer MUST NOT be given signing access to the operator's private key. An
operator-approved signing key alone is still not proof of honest execution.

## Private operator-signed authorization

The JSON object must contain EXACTLY the following fields:

* `schema`: `exp053-operator-run-authorization-v1`.
* `runner_execution_id_sha256`: SHA256 of the opaque execution ID from the
  separately authorized scorer attestation.
* `execution_attestation_sha256`: SHA256 of the complete already-verified
  Ed25519 scorer execution attestation.
* `packet_sha256`: SHA256 of the exact approved human-review packet.
* `score_archive_sha256`: byte SHA256 of the verified private score file.
* `operator_custody_reference_sha256`: SHA256 reference to privately
  retained independent, physical operator review evidence.
* `sequence`: positive INT, monotonically exactly next in operator journal.
* `previous_claim_sha256`: SHA256 of the previous accepted whole signed
  operator authorization, or exactly 64 ASCII zeros for sequence 1.
* `operator_public_key_b64`: canonical 32-byte Ed25519 public key.
* `signature_hex`: Ed25519 signature over canonical sorted JSON of all
  preceding fields (not including this signature field).

The authorization resides in a PRIVATE regular file, at most 16 KiB, mode
0600, no symlinks/hardlinks, no extra/duplicate keys, no NaN/Infinity. Require
operator key and scorer key to be distinct. Validate separately pinned
operator root and all scorer/packet/score receipts BEFORE any claim mutation.

## Single-use operator journal

Exact on-disk private JSON: `{"schema":"exp053-operator-claim-state-v1",
"last_sequence":0,"last_claim_sha256":"000...000"}` for genesis, with
64 zeros in the genesis claim. Real journal is created and independently
reviewed by the human operator, NEVER auto-initialized by research code.

After all ordinary admission checks, private score+retry integrity checks,
statistical analysis, and final input hash checks succeed, acquire an
exclusive OS `flock` on the pre-existing operator-owned journal. Refuse
symlinks, hardlinks, wrong permissions/owner, non-regular files, file
replacement, malformed JSON, unexpected fields, journal truncation, and
journal size >4 KiB. A subsequent authorization is consumed only if its
`sequence == last_sequence+1` AND its
`previous_claim_sha256 == last_claim_sha256`. Persist the new sequence
and SHA256 of the **complete signed authorization bytes** through the SAME
locked file descriptor, fsync before releasing. No automatic backup repair,
rollback or state reset; an interrupted write corrupts closed until the
independent operator manually audits/restores it.

Two concurrent consumers of the same signed authorization must not both
succeed; a second replay attempt must HOLD. If a claim succeeds but the later
public aggregate output fails, the authorization remains consumed (fail
closed; operator may independently issue a new sequence after audit).
Future aggregates may include ONLY the sequence and SHA256 of the accepted
operator authorization—never private reviewer identities, score rows or
operator key material.

## Claims and limitations

This is a **software single-use journal**, NOT a cryptographically secure,
remote replicated monotonic counter. Its protection assumes a separate
operator-controlled filesystem, no rollback of the journal, no forged
operator key, and independently examined native model/residency/process
evidence. A privileged attacker who restores a journal snapshot can
replay an old chain; an operator with the key can also falsely attest.
Thus an independently maintained external high-water/custody witness
is necessary before interpreting these controls as scientific replication
proof. This phase does not create human approvals, execute or deploy models,
modify NAS/Neo4j, or promote any performance claim.
