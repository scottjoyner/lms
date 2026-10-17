# EXP-053 human enrollment and blinded review authority — HOLD

This is a **manual owner-approval workflow**, not an automated human reviewer
enrollment or permission to score a model. As of 2026-10-17, there are
**ZERO** enrolled/verified human reviewers and ZERO admitted case reviews.

## What the owner must do outside the agent

A real human study owner must confirm two *different real people* and their
unconflicted independent participation. Document the verification and consent
privately, ensure that each person independently controls their own Ed25519
signing key, and keep them blinded to each other's answer selections,
the sealed exam answer key and all model predictions. Owner checks may be
retained in an independent private record; its SHA256 appears in the roster.
A third adjudicator, if needed, must be independently vetted, blind and
registered ahead of their decision.

The owner—not ChatGPT, a script, or a test fixture—then writes a **PRIVATE**
mode-0600, non-symlink JSON roster **outside the public repository**:

```json
{
  "schema": "exp053-owner-reviewed-roster-v1",
  "packet_sha256": "<sha256 from independently checked private packet>",
  "owner_custody_reference_sha256": "<sha256 of separate private owner identity/consent evidence>",
  "blind_assignments_confirmed": true,
  "slots": {
    "A": {
      "reviewer_id": "<approved-pseudonym-A>",
      "role": "independent",
      "public_key_b64": "<raw-32-byte-Ed25519-public-key-base64-A>"
    },
    "B": {
      "reviewer_id": "<approved-pseudonym-B>",
      "role": "independent",
      "public_key_b64": "<raw-32-byte-Ed25519-public-key-base64-B>"
    }
  }
}
```

Slot `ADJ` can be added with `"role": "adjudicator"` and a distinct verified
person/key. All key strings above are placeholders, NOT enrollment evidence.
Never put names, contact details, consent documents, private keys,
reviewer full forms, RACE text, signed forms or score archives in public Git.

After independent verification, the owner calculates SHA256 of the **exact
roster file bytes** and requests a reviewed source-code change to pin it in
`research/exp053_race_review.py` as
`APPROVED_REVIEWER_ROSTER_SHA256`. This value must stay empty until manual
authorization. The pinned public SHA is not identity evidence by itself:
the owner must retain out-of-band proof that the people and keys were verified.

The gate takes `--trust-roots <private-owner-roster.json>` for historical
CLI compatibility; arbitrary old `{"reviewer_id":"key"}` maps are now invalid.
In `verified_admission`, reviewer_A must be signed by the A-slot signer,
reviewer_B by the B-slot signer, and any adjudicator by the pre-enrolled
ADJ-slot signer. Different roles, swapped slot IDs, identical keys, unregistered
adjudicators, world-readable/symlinked roster files, unpinned content and
private-data fields are refused.

Do not finalize the cohort until **at least 40 distinct articles** have
genuine independently signed and adjudicated approvals. Completed form
signatures are NOT proof that a person actually read an article; review
quality still requires authentic independent review and audit.

## Independent checkpoint admission still separate

The real three-checkpoint, 39-shard weight-byte inventory was reproduced
39/39 using a second SHA256 implementation; it is preserved at
`EXP053-WEIGHT-IDENTITY-2026-10-16.json`. Its public archive checksum
`e95c771517869f2060b84c7a270bc3e59bd659500c5187d2227d2f94d00440b6`
remains **UNAPPROVED** for score admission. Separately review the actual
weight-file custody and pin only after explicit operator approval.
`WEIGHT_REPORT_SHA` remains empty. The two approvals are distinct.

No EXP-053 neural forward experiment, deployment, model promotion, public
RACE publication, merge, production data write or fabricated signoff is
authorized by this document.
