# EXP-053 — source-free human reviewer owner readiness handoff

**October 19, 2026. NO_GO_RESEARCH_ONLY.** This is a read-only inspection
of public metadata and PRIVATE FILE METADATA ONLY. It does not authenticate
human participation, source review work, signer keys, model bytes, or the
custodian's independence. No licensed corpus passage or private review content
was opened, copied, displayed, included in the output or committed.

## Live observed status (local x1-370)

| Gate | Observation | Meaning |
|---|---|---|
| Public study manifest | 64 distinct pinned candidate articles | Frozen cohort only |
| Minimum independent human sample | 40 distinct co-reviewed items | Each must have two independently signed blinded reviewers |
| Minimum first-pass human work | 80 independently completed case decisions | Disagreements can require more adjudication |
| Frozen manifest human_reviewed_articles | 0 | Original static value, not a live private-file counter |
| Reviewer owner pin | EMPTY | Zero reviews can be admitted by the approval gate |
| Private source packet metadata | PRESENT | Contents and provenance **not** verified by this preflight |
| Private sealed answer-key metadata | PRESENT | Must stay invisible to both blinded reviewers |
| Blinded reviewer forms A / B metadata | Both PRESENT | Empty/filled contents **not** inspected; not review completion |
| Hugging Face source Parquet path | SYMLINK_PRESENT_TARGET_UNVERIFIED | Normal cache layout is plausible; actual target/source hash not read |
| Scientific execution authorization | FALSE | No model inference, promotion or deployment |

All four private review artifacts were observed as owner-owned private regular
files under an owner-private directory, using `lstat` only. This does not
establish RACE text integrity, whether any forms have been filled, reviewer
identity, native human reading, provenance, consent or review quality.

## The smallest real human handoff

1. The accountable study owner designates two genuinely different,
   independent people for blinded slots A and B, confirms their willingness
   and conflicts independently, and controls a **private** identity/consent
   and key-custody record. Never record names, email, private keys or passage
   text in this public repository.
2. The owner privately verifies each human's Ed25519 signing-key possession,
   keeps the reviewers blinded from one another and the sealed exam answers,
   then independently approves and pins a PRIVATE 0600 roster, following
   `EXP053-INDEPENDENT-REVIEWER-ENROLLMENT-2026-10-17.md`. An agent cannot
   create an owner signoff or invent reviewer identities. The owner pin
   remains EMPTY until this happens.
3. The owner verifies the original RACE licensed-source SHA256 through the
   existing source gate in an approved private environment, and provides
   the private blinded packet to each reviewer through an appropriate
   controlled noncommercial research workflow. The offline review HTML
   `research/exp053_offline_review.html` displays one passage at a time;
   the sealed answer key must never be distributed to reviewers.
4. Independently collect completed answer/foil decisions, exact supporting
   source quote and rationale for each accepted item, with reviewer-signed
   receipts. Cover the same >=40 source-distinct cases in both independent
   blinded slots; use an independently enrolled third human to adjudicate
   disagreements. Do not count test signatures, reused human reviewers or
   multiple keys from a single automated agent as independent review.
5. ONLY after this human action, separately run the existing private
   `exp053_race_review.py` admission process to verify reviewed roster, exact
   source/answer key, signatures, 40-item threshold and adjudication. Never
   change a manifest to claim reviews before receipts and independent audit.
6. The external durable checkpoint, independent witness signing authority,
   native negative tests, pinned scorer/runtime/model bytes and other science
   gates are **separate and still blocked** even if human labeling eventually
   succeeds. Scientific execution remains NO-GO.

Reproduce the SOURCE-FREE check:

```
python3 research/exp053_review_owner_readiness.py
python3 -m pytest -q tests/test_exp053_review_owner_readiness.py
```

The CLI ALWAYS exits 2 and reports `scientific_execution_authorized=false`.
Its report does not contain licensed passages or private filesystem paths.
Do not call this output an owner authorization or human adjudication. No
network connection, source text transfer, reviewer/key enrollment, GitHub
publication, model-forward run or production mutation was attempted.
