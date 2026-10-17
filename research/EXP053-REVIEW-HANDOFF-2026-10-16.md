# EXP-053 human adjudication handoff — model-forward HOLD

**2026-10-16: No EXP-053 neural forwards executed; no human labels or signatures obtained.**
Prior EXP051/052 240 synthetic passes remain historical, not the EXP053 scientific evaluation.

## Evidence and privacy

- Research worktree: `/home/scott/git/wt-lms-exp053-adjudication-20261016` on `x1-370`.
- Private reviewer source text, RACE passage packet, unsigned forms, and sealed human-exam answer key: `/media/scott/SSD_4TB/exp053-private-20261016/` (user-local SSD, not NAS).
- Public metadata only: `research/EXP053-COHORT-MANIFEST-2026-10-16.json`; 64 independently identified candidate articles, SHA-256 checksum `43e8309ac1c8ebdb28ae1c478254d43313449e0a44588c301f0a984a240d92b9`.
- Usage restriction: RACE is available for **non-commercial research only**. Never redistribute `packet.json`, source passage text, source-supplying review forms, answer key, or derivatives via GitHub, public artifacts, chats, or unrestricted model training.
- Source is human-authored exam questions, not proven unseen in model pretraining, and not adjudicated for our study until independent reviewer forms are signed.

## Independent review actions

Each of two distinct human reviewers should receive the `packet.json` and **only their own** `reviewer_A.json` or `reviewer_B.json`. Do not show them `sealed_exam_key.json`, other reviewers' forms, model outputs, existing gold labels or signed trust keys.
Review all 64 cases. For each, either reject with a substantive reason or choose exactly one supported answer and one genuinely plausible unsupported foil, cite an exact 20+ character substring from the provided article as evidence, and provide a 15+ character explanation why the foil is unsupported. Keep identities separate.
Set `reviewer_id` to your registered signer ID, set `attestation` to exactly:
`I independently reviewed these source passages and responses; no model predictions or other reviewer labels were provided to me.`
**The assistant cannot supply human signoffs.** Test keys, self-reviewed labels, guessed evidence spans or copied reviewers are invalid evidence.

## Signing and admission

Each reviewer must separately create and safeguard their own Ed25519 PEM key, e.g. `openssl genpkey -algorithm Ed25519 -out reviewer-key.pem && chmod 600 reviewer-key.pem`, outside the repo. Do not commit, share, or expose a private key.
From the worktree, use the isolated Python environment `/home/scott/git/.latent-exp053-data-venv/bin/python`. The script `research/exp053_sign_review.py public-key --private-key <reviewer-key.pem>` prints a raw public key for entry in a private `trust_roots.json` map under the corresponding signer ID. Keep that file private; never commit reviewer identity details.
After **manual completion** of every item, sign using:
`python research/exp053_sign_review.py sign --review <completed-review.json> --packet <private/packet.json> --private-key <reviewer-key.pem> --out <reviewer.signed.json>`.
Preserve the originals separately. Place the two independently signed final reviews into the gate's private directory as `reviewer_A.json` and `reviewer_B.json`.
Then execute:
`python research/exp053_race_review.py gate --private-dir /media/scott/SSD_4TB/exp053-private-20261016 --trust-roots <private/trust_roots.json> --out <private/admitted.json>`.
Disputed reviews **require a separately signed, independent third adjudicator** and `--adjudicator-file`; fewer than 40 accepted distinct articles remains HOLD.

## Audit and next experiment

Use `PYTHONPATH=research /home/scott/git/.latent-exp053-data-venv/bin/python -m unittest discover -s tests -p test_exp053_human_review_gate.py -v` for fail-closed gate tests. These are simulated software tests, **not** human adjudication.
Only after a verified admission receipt is pinned and the reviewed cohort is frozen should a separate, preregistered **EXP-053B** model-forward script be implemented or executed. Paired question×evidence interaction, true foil rejection, free generation, abstention, source specificity, and article-cluster uncertainty remain unmeasured.
No model weights, production routing, Neo4j, NAS, training, merging, deployment or arXiv changes.