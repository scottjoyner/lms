# EXP-053 human adjudication handoff — model-forward HOLD

**2026-10-16: No EXP-053 neural forwards executed; no human labels or signatures obtained.**
Prior EXP051/052 240 synthetic passes remain historical, not the EXP053 scientific evaluation.

## Evidence and privacy

- Research worktree: `/home/scott/git/wt-lms-exp053-adjudication-20261016` on `x1-370`.
- Private reviewer source text, RACE passage packet, unsigned forms, and sealed human-exam answer key: `/media/scott/SSD_4TB/exp053-private-20261016/` (user-local SSD, not NAS).
- Public metadata only: `research/EXP053-COHORT-MANIFEST-2026-10-16.json`; 64 independently identified candidate articles, SHA-256 checksum `43e8309ac1c8ebdb28ae1c478254d43313449e0a44588c301f0a984a240d92b9`.
- Usage restriction: RACE is available for **non-commercial research only**. Never redistribute `packet.json`, source passage text, source-supplying review forms, answer key, or derivatives via GitHub, public artifacts, chats, or unrestricted model training.
- Source is human-authored exam questions, not proven unseen in model pretraining, and not adjudicated for our study until independent reviewer forms are signed.

## Offline browser review (no JSON hand-editing required)

Use the research-only, source-free HTML file `research/exp053_offline_review.html` in a modern offline-capable browser on the reviewer's own device (opening it as a local file is sufficient, with no HTTP server). Select the privately distributed, **blinded** `packet.json` and *only that reviewer's* `reviewer_A.json` / `reviewer_B.json`. The form checks the source packet's SHA-256 in the browser, displays one question at a time, supports precise source quote selection, tracks completion, and exports unsigned JSON as a **local browser download**. It does not transmit or persist the passages in localStorage, make network requests, sign for the reviewer, or supply answers automatically. Browser CSP prohibits network connections. The original sealed exam answer key must be withheld entirely.

A reviewer may export partial drafts and reload them on their own machine, but cannot pass the signing workflow until all 64 decisions have been completed and they affirm the independent-review attestation. Use secure person-to-person custody of licensed packet data; never commit packet text, forms, or output to public GitHub. Browser form code is audited with a JS syntax check and canonical SHA-256 compatibility test against the actual private packet.

**Cryptographic caveat:** Ed25519 verifies possession of a private key, **not human identity, human independence or actual review quality**. The project owner must independently verify and preapprove the actual human reviewer identities, verify that each controls their separate key, and retain a provenance/consent record outside the signed JSON. Never accept generated keys + automated filled forms as human review. Different public keys are necessary but not sufficient evidence of distinct independent reviewers.

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
Disputed reviews **require a separately signed, independent third adjudicator** and `--adjudicator-file`. The third form must contain exactly the disputed IDs, not automatically label all 64. The adjudicator independently examines the same source passages without seeing original answer keys or reviewers' labels; their signed ruling handles every dispute. Fewer than 40 accepted distinct articles remains HOLD.

## Preliminary source-only contamination control

Source overlap audit: `research/EXP053-SOURCE-OVERLAP-AUDIT-2026-10-16.json` (generated via `research/exp053_source_overlap_audit.py`) found **64 distinct full RACE passages**, with **0 exact full-passage matches and 0 shared contiguous 20-token source windows** against 13,594 QASPER source paragraphs and 1,000 QuAC source passages. This is a public, de-identified **source-only** observation; **SQuAD was not available for this audit**, and no result demonstrates absence from any language-model pretraining dataset. Do not overstate it as contamination exclusion.

The model-free analysis supplement, `research/EXP053-MEASUREMENT-PLAN-2026-10-16.md`, was frozen on 2026-10-16 prior to human approvals / EXP053 model forwards, SHA256 `5239be2dd2413af98759a2ac5fdde100f2407d2de07149347f272d1e5ee0922d`. The pure analysis code `research/exp053_metrics.py` refuses changed plans and incomplete article/model/arm/rotation grids. Its successful unit tests use **fabricated synthetic scores** solely for math and safety verification; the 3-model scientific result remains UNMEASURED.

## Audit and next experiment

Use `PYTHONPATH=research /home/scott/git/.latent-exp053-data-venv/bin/python -m unittest discover -s tests -p test_exp053_human_review_gate.py -v` for fail-closed gate tests. These are simulated software tests, **not** human adjudication.
Only after a verified admission receipt is pinned and the reviewed cohort is frozen should a separate, preregistered **EXP-053B** model-forward script be implemented or executed. Paired question×evidence interaction, true foil rejection, free generation, abstention, source specificity, and article-cluster uncertainty remain unmeasured.
No model weights, production routing, Neo4j, NAS, training, merging, deployment or arXiv changes.


## 2026-10-16 supplementary tokenizer and verified-replay readiness

The prospective tokenization plan was frozen BEFORE tokenizer loads:
EXP053-TOKENIZER-PREFLIGHT-SPEC-2026-10-16.md, SHA256
92643aa9e0603d29eeab8dfd24680615036b7ee49d88e12d439a8e5c569b8fb7.
Only 64 candidate passages were used. All three LOCAL tokenizers passed 384
prompts and 1,536 label-token boundary checks EACH (1,152 prompts and
4,608 checks total), using no model weights or neural forwards.
The reproducible source-free summary is
EXP053-TOKENIZER-PREFLIGHT-2026-10-16.json, SHA256
8e35a44e1ebb725c0b52469baa4ab5d41472e38fe55346352b96dd26697726be.
Maximum prompt lengths: K2 887, Qwen 914, MiniCPM 914 tokens.
These checks do not measure semantic grounding, generation, or abstention.

Future aggregate-only report replay is implemented by
research/exp053_verified_analysis.py. It REQUIRES fresh verification of the
independent signed human admission BEFORE reading any private score archive.
It checks the scorer's full prompt hash and option order against the actual
private source, original blinded article, exact question, approved-only decoy,
and the frozen tokenization receipt, then applies strict article-level metrics.
Score archives must remain in the private research SSD folder, and only
source-free aggregate metrics may be committed to public GitHub.

A model-forward runner has NOT been implemented or executed for EXP053.
No signed independent human reviews are available as of this checkpoint.
The apparent success of synthetic unit tests cannot substitute for external
adjudication, nor can possession of a signing key establish human identity.
The operator must independently validate reviewer identities and custody.
The normal GitHub CI runs Python 3.10/3.11/3.12, Ruff and pytest; the suite's
local source/crypto integration cases skip when private dependencies are absent.
