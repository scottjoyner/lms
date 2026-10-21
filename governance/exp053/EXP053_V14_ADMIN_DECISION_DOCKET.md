# EXP-053 — administrator policy decision docket (V14)

**Proposal only; no administrative approval is implied.** Repository: [scottjoyner/lms](https://github.com/scottjoyner/lms). Tracks [governance #47](https://github.com/scottjoyner/lms/issues/47), [draft governance PR #48](https://github.com/scottjoyner/lms/pull/48), and the separate [scientific issue #31](https://github.com/scottjoyner/lms/issues/31).

## Verified before requesting an administrator decision

- As of the V14 handoff (2026-10-20 America/New_York / 2026-10-21 UTC), GitHub connector read-back returned **no repository rulesets** (`[]`) and `main.protected=false`. PR #48 was draft and unmerged, with **zero requested/submitted reviews**.
- A current **GET-only** Fleet Commander preflight passed on 12 observed target branches, with **zero writes**. Earlier in the same session, anonymous REST calls were temporarily blocked by the GitHub API's **60-request/hour unauthenticated quota**; rerun later succeeded after quota reset. A failed rate-limited scan is *not* evidence of a permission denial or protection state.
- The candidate policy is SHA-256 pinned as `461c6bd54985f4c1b0523788d3dd56216ff061b8f27a9c5136f561d781012efd`; baseline evidence is pinned as `dfc303135946a57064597b9097c80cdcbdb471959e6afd61d92bf45113a6a599`. **Neither file was changed in V14.**
- Five check contexts observed in prior successful CI were selected for the **proposed, not-yet-enforced** policy, all from GitHub Actions app ID **15368**: `lint-and-test (3.12)`, `focused-tests (3.12)`, `qualification-tests (3.12)`, `operational-failure-injection (3.12)`, `prompt-cache-tests (3.10)`.
- The public GitHub repository metadata exposes administrator permission on the connected account, **but the connected GitHub tools expose no ruleset-create action** and Fleet `gh` has not been established as authenticated. A connected permission flag is **not** a credential to operate the administrator CLI.

## Five explicit administrator decisions — no defaults inferred

The accountable administrator must record **ACCEPT / REVISE / REJECT** and rationale for each item **outside autonomous tools** before any creation:

1. **Target scope:** protect `main` plus each flat `research/exp053*` branch, with no excluded refs. Nested names are **not covered** by that glob. Confirm the exact preview of all 12 currently observed refs and intended future naming.
2. **Human review gate:** require two genuine **non-author** GitHub approvals per target PR; dismiss stale approvals on pushes; require latest-push approval and resolved conversations. Name the independent eligible human reviewers privately and confirm their willingness, expertise and actual access. An approval from a bot or assumed identity is not evidence.
3. **Status checks:** require the five exact contexts and GitHub Actions app ID `15368`, with `strict_required_status_checks_policy=true`. Strict mode may require rerunning CI when the base branch changes. Do not infer readiness from green checks on a different SHA.
4. **Branch creation exception:** the unchanged candidate has `do_not_enforce_on_create=true` **for the required-status-check rule**. Decide explicitly whether this exception is acceptable; any revision requires a new JSON hash, fresh test evidence and renewed administrator review.
5. **Bypass and emergency recovery:** proposed `bypass_actors=[]`, plus force-push and deletion denials. Confirm GitHub plan/support and whether administrator/emergency pathways can bypass protection; document an accountable break-glass, independent approval and audit procedure **without changing the ordinary no-bypass intent**. If unsupported or risky, do not activate this candidate.

A decision to **REVISE** any material rule means the immutable candidate is superseded; the existing guarded installer must reject it until its policy bytes, pinned hash and tests are intentionally updated under independent review. Do not approve the installer by leaving a comment that says only "continue".

## Authorized implementation order — stop at each failed gate

1. Administrator checks [Settings → Rules → Rulesets](https://github.com/scottjoyner/lms/settings/rules), repo plan support, visible inherited rules and any legacy branch protection with authenticated privileges. If an overlapping rule exists, **stop** instead of replacing it automatically.
2. In the isolated checkout, run the **offline** policy validator and guarded helper default mode (both zero-write). Then run the optional **read-only** current-head preflight. If that reports `RATE_LIMITED_READ_ONLY`, wait for quota restoration or use an independently authorized GitHub token in the operator's secure environment; **do not embed a token in a command, report, issue, or repo**.
3. After documenting all five decisions and verifying credential provenance, the human administrator may choose **one** manual GitHub UI activation or separately authorized single-create API action. No assistant/agent automatic activation. Avoid retrying a create request with uncertain outcome. **V17 pre-POST race guard:** the guarded installer re-enumerates active/inherited rulesets and the exact repository branch roster *after* its full current-head CI scan, just before its single eligible POST. A newly installed ruleset or changed branch roster causes a fail-closed no-write result. V18 also retains the exact HEAD SHA for every branch whose source-bound CI passed, then rereads every targeted branch's HEAD and unprotected status immediately before POST. An advancing HEAD, a changed protection flag or a missing target fails closed even when the target-name roster is unchanged. These final GETs narrow, but cannot eliminate, a competing administrator's last-moment GET-to-POST race. In case of a timeout or uncertain POST result, do not retry; inspect the authenticated repository policy inventory and effective branch rules manually.
4. Independently inspect the active rule detail **and effective per-branch rules**. Use the GET-only `exp053_verify_live_enforcement.py`; metadata check must report `POLICY_METADATA_VERIFIED` with no unverified bypass. A named ruleset listing alone is insufficient.
5. An independent human uses the V9–V13 [negative merge-denial protocol](EXP053_V9_NEGATIVE_MERGE_CHALLENGE_HANDOFF.md): two otherwise mergeable **non-draft** harmless fixture PRs, one against `main` and one representative protected research branch, with passing required checks. Capture GitHub's *specific lack-of-human-approvals* refusal, without invoking Merge or enabling auto-merge. Capture current SHA, observed ruleset ID, human identity, screenshots/evidence digest in separately controlled storage. Draft PRs #37–#46 **cannot** serve as negative challenges.
6. Nominate and request actual human review on exact draft PR heads only when the independent review gate exists. Keep the draft PRs unmerged until genuine approvals and policy-readback evidence are recorded. Any merge must be a separate authorized maintainer decision.

## V23 — make a pinned required check execute the governance suite

**Previously discovered gap:** the separate `EXP-053 Governance Offline` workflow is
path-filtered and its `governance-offline` job is **not** among the five status
contexts in the immutable proposed ruleset. Those five checks could have
succeeded even if the governance-only workflow failed.

The draft changes the existing **required** `CI / lint-and-test (3.12)` job:
after its regular test suite, when `governance/exp053` is present in the PR
checkout, it runs the full offline `test_exp053_*.py` unittest discovery,
pinned policy candidate validator, and **zero-write** administrator dry run.
The normal shell exit code gates the existing `lint-and-test (3.12)` check:
a governance test failure fails that already-proposed required check.
This avoids making the path-filtered governance job a sixth required check
that could be absent on unrelated changes.

**Scope and limitations:** the source workflow itself must be present on the
target protected branch after a separately authorized merge; this draft alone
does not change the checks configured on `main` or the existing research
branches. The step deliberately skips the governance gate on a checkout where
the governance directory has *not yet been adopted* (as on baseline `main`);
deleting or altering workflows/files in a later PR remains a possible bypass
subject to independent human review, not a tamper-resistant external check.
An actual installation still requires administrator acceptance, active
effective rules, an independent negative merge challenge, and human reviews.
Neither candidate JSON nor its SHA-256 pin is changed.

## V24 — fail closed on governance-suite deletion after adoption

V23 made the existing proposed required `CI / lint-and-test (3.12)` check run
the offline governance tests **if** the `governance/exp053` directory existed.
A PR removing the directory could otherwise skip all governance tests while
keeping that required check green.

V24's required CI step uses a two-parent-depth checkout and inspects
`HEAD^:governance/exp053` (the protected target's first parent in an ordinary
GitHub `pull_request` test-merge checkout). When the **base already has** the
governance tree, deleting the tree in the proposed head now explicitly **fails
the required job**. Retaining the tree runs all `test_exp053_*.py` tests,
candidate validator and zero-write default administrator preflight; creating
a partial tree without the activation module also fails. No first-parent Git
object means fail closed instead of silently assuming pre-adoption.

This is a **bootstrap-aware, in-repository** protection only. Before
`governance/exp053` exists on the target base, unrelated PRs may still
skip this suite to avoid blocking initial rollout; an authorized merge must
first adopt the files and updated CI into `main` and subsequently every
relevant research base as applicable. The workflow itself is editable via
source PRs, so an actual two-human ruleset and independent reviewer scrutiny
remain necessary and no claim of tamper resistance is made. GitHub Actions
checkout behavior, real target/base ancestry and negative merge refusal
still require the previously defined physical readback/challenge gates.
The immutable candidate ruleset and all NO-GO statuses are unchanged.

Five new tests in `test_exp053_ci_adoption_gate.py` create disposable local
Git repositories and execute the **real workflow shell block** with stub
offline test/validator/administrator commands. They verify allowed first
adoption, refusal of suite deletion, continued runs after adoption, refusal
of incomplete new suites, and refusal of missing parent history. No actual
GitHub operation, science run or production write is performed.

## V26 — retain adopted governance test modules in the required check

V24 detected complete removal of the governance directory, but Python
`unittest discover` exits successfully when **zero tests are discovered**.
An attempted deletion of individual `test_exp053_*.py` files could therefore
leave the existing required `lint-and-test (3.12)` job green, even after
governance tests had been adopted on the target base.

The draft required job now requires at least one local `test_exp053_*.py`
module wherever the governance directory exists. If the target/base parent
already contains governance, the job inventories every `test_exp053_*.py`
module from its exact Git tree and refuses to proceed if **any** adopted
test file is absent from the candidate checkout. It also refuses an
already-adopted base with no tests. Three additional disposable Git-history
tests cover removing the last module, removing only one of two adopted
modules, and adopting a new governance directory without test files.
The test suite is still discovery-based, so new test modules are included.

This checks **module presence**, not immutable module contents: source files,
tests and the CI workflow itself remain changeable by a source PR. The
protected branch must first adopt the CI workflow, and independently
authorized reviewer approvals and effective live GitHub rules remain required.
The candidate policy hash is unchanged; V26 grants no admin or research
authorization and performs no GitHub settings write or merge.

## Release/HOLD matrix

| Gate | Current observation | Required evidence to change decision |
|---|---|---|
| Offline candidate integrity | PASS | Immutable JSON/evidence digests; independent review of any revision |
| Current-branch read-only CI preflight | PASS at V14 observation, non-atomic | Fresh readback on same branch/commit before activation |
| Active effective GitHub branch protection | **NO-GO** | Authenticated ruleset + effective-branch readback |
| Real negative unapproved merge denial | **NO-GO** | Independent human witnessed, non-draft, conflict-free, passing-CI refusal |
| Genuine code reviewers | **NO-GO** | Two approved human identities, access, distinct approvals on exact SHA |
| EXP-053 science authorization | **NO_GO_RESEARCH_ONLY** | Separate blinded source review, ≥40 genuinely independently adjudicated distinct articles, weight-identity pins, independently controlled replay-resistant high-water and physical witness |

**Unconditional boundary:** This docket authorizes **no** ruleset mutation, merge, reviewer fabrication, credential enrollment, RACE material processing, signer registration, inference or deployment. `NO_GO_GOVERNANCE` and `NO_GO_RESEARCH_ONLY` remain until independent, affirmative evidence satisfies the separate gates.
