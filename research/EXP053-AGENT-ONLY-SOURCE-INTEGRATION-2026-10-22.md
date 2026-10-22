# EXP-053 source-only security integration — automated CI, zero GitHub reviews

**Scope:** code, synthetic tests, HTML review UI and public provenance notes
copied verbatim from the public EXP-053 source-security chain through commit
`e74a6f1492973cfbf915578a30cba051aadbd2dd` (#32 and #37–#46) onto
`main` as a new source-only PR. This change intentionally does **not**
cherry-pick the 95-commit experimental lineage, historical results for
EXP-051/052, or unrelated workflow changes.

**Source integrity:** the 128 selected public files were verified byte-for-byte
identical to their original source Git checkout before this document was added.
No existing `main` files were overwritten.

**Automated code gate:** on xwing's isolated temporary worktree, all
`tests/test_exp053*.py` reported **401 passing tests and 276 passing
subtests** (2026-10-22) without using private source data, calling a model
or requesting a human reviewer. Exact hosted CI must pass again for the new PR.

**Owner authorization:** no human GitHub reviewer, independent code sign-off,
review request, or manual oversight is required for the ordinary code merge.
The installed GitHub policy enforces zero required approvals and pinned CI.
Autonomous agents may continue source-only development, tests and merge
preparation. They must not fabricate human adjudication or independent
witness signatures to pass any legacy scientific assertions.

**Science/claims:** this is a security-source integration, **not** validated
K2/Qwen/MiniCPM model-grounding evidence. Experiments, confidential corpora,
external attestations and model release must carry accurately labeled
machine-generated or independently sourced evidence and comply with licenses.
Never label agent-generated adjudication as blind human adjudication. Research
issue #31's existing scientific study is not automatically completed by
merging this code.

**Safety properties:** no changes to production runtimes, model weights,
NAS/Neo4j, secrets, GitHub rulesets or other agent services in this patch.
