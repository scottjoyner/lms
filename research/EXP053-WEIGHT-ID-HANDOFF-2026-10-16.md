# EXP-053 checkpoint identity verification and reporting HOLD

## Purpose and completed source-free work

The weight-file identity protocol was frozen before a full checkpoint hash scan:
`EXP053-WEIGHT-ID-PREFLIGHT-SPEC-2026-10-16.md`, SHA256
`b89f573acc2567b114e65746b3026e36636aee9add6c6568c0fac6319f620a81`.
The read-only scanner is `research/exp053_weight_identity.py`.

The scanner validates the frozen tokenizer/config digests, exact weight shard
filenames, inventory completeness, resolved symlink locations, per-file SHA256
and byte counts, and stable canonical digests. The optional actual scan requires
`--execute-read-only-hashes --out <new source-free research receipt path>`.
Do not use model loading, inference, GPU acceleration, production serving,
network transfers, or modified model weights as substitutes for this read-only
identity proof.

**2026-10-17 update:** The actual read-only multi-gigabyte checkpoint scan
subsequently COMPLETED. See
`EXP053-CHECKPOINT-CUSTODY-AUDIT-2026-10-17.md` and the source-free
`EXP053-WEIGHT-IDENTITY-2026-10-16.json`, SHA256
`e95c771517869f2060b84c7a270bc3e59bd659500c5187d2227d2f94d00440b6`.
The system `sha256sum` command independently reproduced **39/39 weight shard
digests**. This is verified local-file identity, NOT automatic source approval,
model evaluation or proof of an actual inference runner's behavior.

## Fail-closed reporting handoff

`research/exp053_verified_analysis.py` now requires a separately reviewed
and pinned weight-file receipt before scientific aggregate reporting can read
the private score archive. `WEIGHT_REPORT_SHA` is **deliberately empty**.
The gate therefore refuses all current score reports, even with apparently
well-formed checkpoint identity values. Never fill it with a test fixture,
an automatically generated guessed hash, or an unreviewed archive.

Future operator sequence:
1. Execute the frozen read-only scanner on the pinned research host and retain
   its exact file-byte inventory. No model forwards are authorized by scanning.
2. Independently review source-custody metadata and the receipt's actual file
   provenance; freeze and approve the resulting digest in a reviewed code change.
3. Separately establish real human reviewer identities and distinct keys,
   collect independently signed human reviews, and admit >=40 different articles.
4. Only after all prior steps and explicit experimental approval should a
   separately frozen EXP053 forward runner generate private row receipts. The
   source-checked analysis tool can then verify and report public aggregates.

This sequence is **not** automatic approval for any experimental model run.
The checkpoint receipt itself cannot prove an execution engine used the weights;
independent runner attestations remain necessary for that claim.
RACE text, answer keys, reviewer forms and private score rows remain private.

Current evidence: the candidate-only tokenizer preflight passed 4,608 boundary
checks over three tokenizers; no native EXP053 forward passes or independently
verified human signoffs exist. PR #32 remains draft; no deployment or publication.

## Human-review trust bootstrap is also explicitly disabled

The actual `verified_admission` function now requires
`APPROVED_REVIEWER_ROSTER_SHA256` to match the supplied private reviewer
public-key roster. That constant is deliberately empty. This prevents
caller-generated Ed25519 keys, arbitrarily self-selected signer IDs, or
synthetic review fixtures from being mistaken for independently enrolled
human reviewers. The intended future step is for the study owner to
verify two distinct human identities and key possession offline, approve
the research-only assignments, and pin the SHA256 of the exact
public-key map in a code-reviewed change BEFORE human reviews are used.

A pinned key-map digest is necessary but not sufficient evidence of
true human independence; the out-of-band reviewer enrollment evidence
and audit still matter. These checks remain closed to agent-generated
or test-generated identities. The owner has not approved any roster yet.

## 2026-10-17 preparation-only integrity followup

The future public aggregate-replay verifier now **rejects unexpected fields**
at report/model/shard level, invalid or noncanonical lowercase SHA256 digests,
unsafe or repeated shard filenames, non-integer/boolean byte counts, unordered
shards, mismatched total weights, and inappropriate shard-index presence.
These are structural anti-corruption checks, not proof of authenticity by
themselves. As of 2026-10-17 the actual read-only weight scan is COMPLETE and
its individual SHA256 hashes were independently reproduced 39/39; the
**owner-reviewed scientific checksum pin is still absent** and
`WEIGHT_REPORT_SHA` remains empty. Software tests patch in **synthetic report
metadata only** and do not sign reviews, score RACE questions, examine tensor
values, or establish semantic-grounding research outcomes.

The offline-only reviewer HTML now captures `input` and `change` events,
updates progress using the in-memory unsigned draft on each edit, and activates
the unload-warning guard before navigation. Clicking the source-quote helper
does the same. It still NEVER persists any private passages, reviewer labels,
or drafts automatically to browser storage or a remote service. A human must
explicitly export a local unsigned JSON draft and independently sign later.
A Node-based DOM fixture tests the unsaved-work alert using synthetic text.

The independent human identity roster and actual checkpoint inventory remain
unapproved, so admission and scientific replay continue to fail closed.
