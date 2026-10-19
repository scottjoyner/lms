# EXP-053 — independently witnessed negative custody challenge plan (research-only)

This protocol is frozen **before implementation**. It makes a deterministic,
source-free, non-executable test checklist from the already frozen v1 custody
claims schema. It never discovers privileges, attempts authentication, reaches
a node, mutates a witness, signs an attestation or grants authority.

Input MUST pass `exp053_custody_boundary_audit.parse`, preserving its 16 KiB
limit, strict schema, frozen protocol hash, no-follow local file read and
absence of privileged/private fields. Output uses exact schema
`exp053-custody-negative-challenge-plan-v1`. Output contains only:
`schema`, `candidate_alias`, `input_sha256`, `decision`,
`scientific_execution_authorized`, `independent_administration_approved`,
`witness_durability_approved`, `all_scenarios_executed`, `scenario_count`,
`scenario_rows`. Each scenario row has EXACT fields `scenario_id`, `domain`,
`actor`, `challenge`, `expected`, `execution_status`, `evidence_status`,
`claim_status`. `input_sha256` hashes exact input bytes, and is only a content
reference, NOT a signature, timestamp, approval or tamper-proof checkpoint.

Use these 12 fixed source-free challenges, in exact order:
1. `SCORER_SSH_ACCESS` in `witness_remote_login` for `scorer_operator`:
   independently check whether scorer can log in to witness context.
2. `FLEET_SSH_ACCESS` in `witness_remote_login` for `fleet_automation`:
   independently check whether fleet automation can log in to witness context.
3. `SCORER_ROOT_CONTROL` in `witness_root_administration` for `scorer_operator`:
   challenge effective root/sudo/console authority, not just key path.
4. `FLEET_ROOT_CONTROL` in `witness_root_administration` for `fleet_automation`:
   challenge effective root/sudo/console authority.
5. `SIGNER_READ_OR_USE` in `witness_signing_key_custody` for `scorer_operator`:
   challenge unauthorized signer access or signing impersonation.
6. `HIGHWATER_REWIND` in `witness_highwater_mutation` for `scorer_operator`:
   challenge CAS regression and replay after local scorer rollback.
7. `SNAPSHOT_RESTORE` in `witness_snapshot_restore` for `scorer_operator`:
   challenge host/hypervisor/snapshot rollback to prior accepted state.
8. `BACKUP_RESTORE` in `witness_backup_restore` for `fleet_automation`:
   challenge restoring witness durable high-water from an old backup.
9. `TRUST_ROOT_REPLACEMENT` in `witness_trust_anchor_rotation` for
   `scorer_operator`: challenge changing or revoking the witness trust root.
10. `CHECKPOINT_FORK` in `witness_checkpoint_publication` for
   `scorer_operator`: challenge editing, deleting, backdating or forking an
   independently published high-water checkpoint.
11. `FLEET_SIGNER_ACCESS` in `witness_signing_key_custody` for
   `fleet_automation`: challenge automation abuse of signer key.
12. `FLEET_CAS_ACCESS` in `witness_highwater_mutation` for
   `fleet_automation`: challenge automation abuse of CAS state.

Expected outcome for every *future independent nonproduction* challenge is
`DENIED_AND_INDEPENDENTLY_VERIFIED`, never a report from the scorer alone.
`execution_status` MUST be `NOT_EXECUTED`, `evidence_status` MUST be
`NOT_ATTESTED`, and `all_scenarios_executed` MUST be `false`. The
`claim_status` is exactly one of `CLAIMED_CAN`, `CLAIMED_CANNOT_UNVERIFIED`,
`CLAIMED_UNKNOWN` and `MISSING` and does not assert actual permissions.

All output must be source-free and print neither evidence digests from input,
actor identities, SSH usernames, IPs, hostnames other than the authorized
alias, key details, private paths, or research text. Malformed data and hash
drift FAIL CLOSED. For valid or invalid input CLI exit is **2**.

Every report MUST say `decision=NO_GO_RESEARCH_ONLY`, with all three approval
flags `false`. A complete 12/12 plan is **not** an executed witness test.
No model inference, signing, witness creation, SSH, network traffic, remote
file or service change, NAS/Neo4j write, deployment, merger or publication is
authorized by this design. Later custodian attestations require separately
reviewed hardware, durable evidence, independently signed receipts and an
entirely separate guarded authority path.
