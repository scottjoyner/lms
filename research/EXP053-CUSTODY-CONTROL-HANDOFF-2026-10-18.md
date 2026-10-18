# EXP-053 independent custody control graph — negative screen handoff

**Research-only NO-GO; no independent custody or scientific approval.**

A prospective, SHA256-frozen specification in
`EXP053-CUSTODY-BOUNDARY-NEGATIVE-AUDIT-SPEC-2026-10-18.md` defines a strict
source-free claims schema, 8 witness-administration domains and 3 roles. The
new `exp053_custody_boundary_audit.py` accepts only a bounded local JSON
file with verified structure. It DOES NOT verify claim signatures, witness
administration, root/sudo privilege, signer key isolation, durable CAS,
revocation, backup control or proof of human review. A digest is only a
reference. Even a fully populated matrix of negative claims is NO-GO.

## Real controller-side Lenovo pilot observation

The source-free example `EXP053-LENOVO-CUSTODY-CONTROL-CLAIMS-2026-10-18.json`
points at SHA256 of `EXP053-PHYSICAL-WITNESS-QUALIFICATION-2026-10-17.md`
(`ce82af3f9df092de502166e46df6adcdc776867e150a3cc654e2ff55378366e7`).
This is an **operator observation only**: existing fleet SSH reached Lenovo.
If Lenovo were selected for future witness engineering, that shared controller
login would remain a separation concern. This does not show access to a
*deployed witness*, sudo, root, signing keys or snapshots; no deployed witness
exists. All such privileges are explicitly UNKNOWN, not benign or absent.

The sample has one claimed shared access domain (`witness_remote_login`),
23 unproven or omitted role/domain cells, and no independently authenticated
custody source. The CLI emits `NO_GO_RESEARCH_ONLY` with exit code **2**.
No original SSH addresses, user IDs, private key paths, public or private keys,
or research passages appear in the fixture, parser output or tests.

## Custodian handoff — evidence to obtain *outside* this repository

1. Have a separately accountable human identify the actual operator, witness
   root, fleet runner, hosting/network admins, backup/restore controllers, and
   independent signer/publication trust administrators. Keep private details
   out of Git; store only independently attested hashes/custody references.
2. Inspect actual effective privilege and restore pathways, including shared
   SSH/agent identities, sudo, remote console, hypervisor, backups, and key
   rotation/revocation. Reject self-asserted separation as proof.
3. In an independently approved *disposable nonproduction* setup, challenge
   scorer access to witness signer, CAS state, journal, snapshots, backup
   restoration, trust roots, and publication. Verify denial by the separate
   custodian, including rollback after reboot/recovery and high-water gaps.
4. Require signed external native evidence plus independent checkpoint
   publication, signature verification, provenance/age limits, and replay
   detection. That is a separate, later approval path; never allow this audit
   to turn on real approval pins or live services.
5. Independently finish >=40 signed human article-disjoint reviews and the
   verified exact runner/checkpoint custody before considering model inference.

**No keys created, no SSH or remote system changes, no external witness
created, no model forward, NAS/Neo4j write, production deployment, or merge.**
