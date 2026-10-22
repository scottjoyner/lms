# EXP-053 V16 — independent human source-review roster and order

Read-only public GitHub inventory on **2026-10-21 UTC**, repository `scottjoyner/lms`. Source tracking: [P0 governance issue #47](https://github.com/scottjoyner/lms/issues/47), [draft policy PR #48](https://github.com/scottjoyner/lms/pull/48), [separate science issue #31](https://github.com/scottjoyner/lms/issues/31).

**Decision remains `NO_GO_GOVERNANCE` and `NO_GO_RESEARCH_ONLY`.** This is a *review-planning* artifact based on current PR metadata and changed filenames, not a code security assessment or human approval. No reviewer identities have been nominated, consented, requested, or verified; none should be inferred from this file.

## Observed exact-head stack

Each PR's base SHA equals the preceding PR's head SHA for #38–#46. These are **incremental stacked diffs**. Review each PR *against its own stated base*, not `main`, and independently recheck the head/base SHA before inviting reviewers.

| PR (all open DRAFT) | Exact head SHA | Base branch | Changed files; +/- lines | Changed source modules | Suggested independent source-review expertise |
|---|---|---|---|---|---|
| [#37](https://github.com/scottjoyner/lms/pull/37) | `8244ece960a93fe02a97a9ed3451581e3c137286` | `research/exp053-race-human-review-gate-20261016` | 12; +1,139 / -22 | `research/exp053_race_review.py`, `research/exp053_reviewer_roster.py` | Ed25519 public-key validation and key-subgroup attacks; manifest canonicalization; reviewer trust/custody |
| [#38](https://github.com/scottjoyner/lms/pull/38) | `68ffb084cec264330c488482ae447b4de063f697` | `research/exp053-owner-roster-read-review-20261019` | 10; +970 / -10 | `research/exp053_race_review.py` | Approval-before-source access, bounded untrusted JSON, race-safe manifest read |
| [#39](https://github.com/scottjoyner/lms/pull/39) | `0a473861fc8306728d3a62ab93919aed3513a356` | `research/exp053-private-admission-source-review-20261019` | 6; +432 / -5 | `research/exp053_race_review.py` | Parquet/pyarrow input decoding, source hashing, file descriptor and TOCTOU safety |
| [#40](https://github.com/scottjoyner/lms/pull/40) | `8ed7a6b06b33efc625acae2f3068805c6986b865` | `research/exp053-source-fd-cohort-review-20261019` | 6; +525 / -4 | `research/exp053_race_review.py` | Private file creation modes, symlink ancestor defenses, filesystem race conditions |
| [#41](https://github.com/scottjoyner/lms/pull/41) | `0de5117178581310e697b2a8e05228f75a56f2a3` | `research/exp053-private-writer-ancestor-review-20261019` | 5; +313 / -5 | `research/exp053_race_review.py` | Sealed case manifest coverage, completeness and fail-closed answer-set validation |
| [#42](https://github.com/scottjoyner/lms/pull/42) | `d96f66cacd8eb3f35644c88ee25de28a0a150550` | `research/exp053-sealed-scope-review-20261019` | 5; +367 / -16 | `research/exp053_verified_analysis.py` | Parsed-byte receipt binding, replay and immutable evidence consistency |
| [#43](https://github.com/scottjoyner/lms/pull/43) | `54330c12718cd910ffe23001783be64d53a544e4` | `research/exp053-replay-pinned-receipts-review-20261019` | 6; +314 / -4 | `research/exp053_race_review.py`, `research/exp053_verified_analysis.py` | Atomic output publication, directory fd/inode binding, filesystem substitution |
| [#44](https://github.com/scottjoyner/lms/pull/44) | `1085a7427eabfd33140fe52d5c3d397f870a1efd` | `research/exp053-aggregate-fd-review-20261019` | 5; +327 / -88 | `research/exp053_operator_run.py` | Operator journal safety, single-use marker, directory ancestor file descriptors |
| [#45](https://github.com/scottjoyner/lms/pull/45) | `afbc1f9090b79c57c1eaff4436b261402cd2a42d` | `research/exp053-operator-journal-ancestors-review-20261020` | 5; +308 / -15 | `research/exp053_execution_attestation.py` | Scorer evidence custody, symlink ancestors, inode substitution, attestation boundaries |
| [#46](https://github.com/scottjoyner/lms/pull/46) | `e74a6f1492973cfbf915578a30cba051aadbd2dd` | `research/exp053-execution-evidence-ancestor-review-20261020` | 5; +240 / -3 | `research/exp053_execution_attestation.py` | JSON exponent overflow/underflow, decimal numeric fidelity, scoring input integrity |

Across these incremental PR diffs, GitHub currently reports **4,935 additions and 172 deletions** (not a unified `main` diff). Every PR also modifies `tests/test_exp053_*.py` or a relevant `tests/test_exp053_*.py` equivalent and contains source-only review notes. Consult GitHub's **Files changed** tab for the exact per-PR paths; do not rely on this summary after a push.

**Reviewer status checked for each PR separately:** author `scottjoyner`; open/draft; unmerged; zero requested reviewers; zero submitted reviews. `git ls-files` in the exact governance worktree found **no tracked CODEOWNERS file**. These facts do not establish that any potential reviewer is independent, qualified, willing, or empowered to approve.

## Practical human review sequence

1. The accountable maintainer privately identifies **at least two distinct actual humans** capable of independently scrutinizing this security-sensitive source, tests and science integrity risks. Verify their GitHub handles, suitable repository access, conflict-of-interest independence, expertise and willingness. No bots, pseudonymous simulated accounts or manufactured approvals. Because all ten PRs have the same author, neither approver may be that author for these approvals.
2. Obtain a recorded, explicit administrator decision on the proposed governance policy and create effective review protection through an authorized administrator interface. See the [V14 administrator docket](EXP053_V14_ADMIN_DECISION_DOCKET.md). The current proposed requirements are two genuine non-author approvals, stale-review dismissal, most-recent-push approval, resolved conversations and five source-bound GitHub Actions checks. A **draft state** or `mergeable: true` alone proves none of these safeguards.
3. After effective protection and a **separate human-witnessed, non-draft missing-approval denial challenge**, begin independent source reviews in upstream-to-downstream order: #37 → #38 → #39 → #40 → #41 → #42 → #43 → #44 → #45 → #46. Check the exact head/base binding before each invitation; changes to earlier branches may invalidate downstream review and CI evidence. No automated `ready_for_review` change or invitation is authorized by this document.
4. Ask reviewers to inspect source changes and negative tests, explain their independent decisions with actual GitHub approvals or requested changes, and resolve discussions. Preserve reviewer custody off-repo where appropriate. **Do not treat any one approval as approval of the entire stack**, or infer reviewer endorsement from green CI.
5. Before any separately authorized maintainer merge, recheck active effective branch rules, actual review count/distinct identities, current head/test-merge CI checks, review freshness, branch-head stability, and any bypass or emergency path. No merge is approved by this document.

## Science and privacy separation

This roster contains **public metadata and source filenames only**. It does not display or access any licensed RACE passage, sealed prompt/answer, operator secret, signer/private key, or real model output; it performs no scoring/inference. Even fully approved and merged source code does **not** authorize EXP-053 scientific admission. The independently adjudicated ≥40 distinct article gate, blinded human source review, owner-approved model/weight identity, off-host high-water witness and physical attestation remain separate under [issue #31](https://github.com/scottjoyner/lms/issues/31).

**Readiness:** reviewer roster **not yet appointed**; effective branch review protection **absent**; independent negative merge-denial observation **absent**. Maintain **`NO_GO_GOVERNANCE` / `NO_GO_RESEARCH_ONLY`**.
