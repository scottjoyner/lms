# EXP-053 machine-only evidence path — no reviewer signatures

**Purpose:** keep autonomous research agents moving when no human reviewers
are available, without misrepresenting automated answer-key labels as
independently human-adjudicated scientific evidence.

## Modes

The new `research/exp053_agent_only.py` defaults to an offline,
source-free, read-only preflight against the existing pinned 64-item public
RACE manifest. It reports `human_review_required=false`,
`human_approval=false`, `model_forward_passes=0`. This preflight requires
no extra reviewer identity, signature, interactive web login, or API key.

An agent with access to the **legally authorized, non-commercial** pinned
dataset and the secure private packet plus sealed answer-key snapshot can
explicitly request a private, no-overwrite proxy receipt:

```bash
python3 research/exp053_agent_only.py   --private-dir /owner-private/exp053-packet   --private-out /owner-private/exp053-agent-proxy-receipt.json
```

The private mode validates the pinned public manifest and source/protocol
hashes, matches the private packet and sealed keys, rejects changes to
the question, passage, answers and option digests, and chooses a
deterministic **wrong exam option** for each of the 64 cases. The foil
is not independently reviewed for plausibility or semantic correctness.
The receipt remains private and is marked
`exp053-agent-answer-key-proxy-v1`, `human_approval=false`,
`agent_only_exploratory=true`. No model-forward calls occur here.

## No fraudulent review receipts

This path **does not** produce an `exp053-admitted-v1` human review
receipt or pretend to satisfy the original blind-annotator study. The
original human-adjudicated code remains unaltered for experiments that
explicitly make those claims. An owner may run a separate
machine-only exploratory comparison using a private proxy receipt,
but results must be labeled *exam-key-proxy exploratory* and must not be
reported as proof of independent semantic grounding.

**No manual GitHub reviewers or oversight are required for code, tests,
preflight, receipt generation or ordinary PR merging.** The installed
ruleset requires zero human approvals while preserving CI and branch
integrity. Source licensing, source identity, privacy and truthful claims
are technical/integrity properties rather than personnel sign-off.
Production model activation, NAS writes, downstream acceptance and
scientific promotion are not performed by this preflight.
