# EXP-053 offline observer publication-chain verifier — October 18, 2026

**NO_GO_RESEARCH_ONLY. No real external observer publisher or independent
checkpoint anchor has been deployed, validated, or approved.**

## This milestone

Added the independently frozen, research-only source specification
`EXP053-OBSERVER-PUBLICATION-CHAIN-SPEC-2026-10-18.md` with exact SHA256
`a387aa42f93b55a8ad1325dbddd5621e906bfc165b23967e42f65e9ebceff5e5`.
Added `exp053_observer_publication_chain.py`, a bounded, read-only verifier
that chains **untrusted self-contained signed observer statements** to the
previously frozen 12-case plan and a current receipt checkpoint reference.
It enforces an explicit candidate+plan binding, canonical JSON, a strictly
contiguous genesis-to-head sequence, monotonic locally screened timestamps,
Ed25519 signature verification, separate caller-supplied public-key fingerprint
and head digest, and exact match between head's reference digest and supplied
receipt history. A wrong or missing parent/signature/key/head/reference refuses.
Existing frozen custody, receipt, cross-bundle and reference comparison gates
must pass before it can report a narrowly local consistency match.

This is one increment beyond `eac67a4`: if a retained original head digest is
presented, a missing signed statement, shorter chain or substituted reference
can now be rejected even when a rewritten local reference and its SHA256 agree.

### Critical counterexample

The synthetic tests also replace the entire observer chain, generated test
public key and BOTH caller-supplied digests. That modified history validates
locally, demonstrating this verifier **does not deliver rollback protection**.
Its report always sets:

- `publication_custody=CALLER_SUPPLIED_UNTRUSTED`
- `durable_replay_protection=NOT_PROVEN`
- `scientific_execution_authorized=false`
- `independent_administration_approved=false`
- `witness_durability_approved=false`
- `decision=NO_GO_RESEARCH_ONLY` (exit status 2 for ALL inputs)

The tool cannot tell who controlled the publisher's private key, whether an
observed checkpoint ever existed outside local memory, whether old signed
statements were omitted or whether the scorer altered both CLI pins. It cannot
verify the native durability of earlier reference snapshots. Locally verified
signatures are NOT independent human or operator approval, and a source-free
synthetic statement is not an actual physical negative test.

## Available evidence and what was not attempted

The existing Lenovo fixture still has zero receipt attestations and an
untrusted empty source-free reference. No observer public key or independently
retained head pin is configured. Consequently there is **no usable real
publication input** and no evidence that even an initial external checkpoint
was published. Test scripts create ephemeral Ed25519 keys only as synthetic
in-memory fixtures; they never save or install fleet witness credentials.

Reproduce the synthetic analysis with:

```
python3 -m pytest -q tests/test_exp053_observer_publication_chain.py
```

**Next real gate:** have a separately accountable custodian establish their
own nonproduction signer, root provenance, custody-separated reference/head
publication, native durable high-water, privileged recovery controls and
independent readback. Human-approved disposable negative tests must show
scorer/fleet credentials cannot replace that root, access snapshot/backup
restoration or reset monotonic sequence across reboot and recovery. Those
private results require out-of-band human approval, not a new controller
CLI flag. Separately, 40+ independently reviewed article-disjoint RACE labels,
scorer exact-source control and checkpoint/model-byte custody remain blocked.

No keys created in production; no remote access, real witness, production
service, model forward, NAS/Neo4j write, GitHub publication or deployment was
attempted. Branch remains local and research-only until separate review.
