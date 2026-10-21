# EXP-053 — administrator policy decision docket (V14)

**Proposal only; no administrative approval is implied.** Repository: [scottjoyner/lms](https://github.com/scottjoyner/lms). Tracks [governance #47](https://github.com/scottjoyner/lms/issues/47), [draft governance PR #48](https://github.com/scottjoyner/lms/pull/48), and the separate [scientific issue #31](https://github.com/scottjoyner/lms/issues/31).

## Verified before requesting an administrator decision

- As of the V14 handoff (2026-10-20 America/New_York / 2026-10-21 UTC), GitHub connector read-back returned **no repository rulesets** (`[]`) and `main.protected=false`. PR #48 was draft and unmerged, with **zero requested/submitted reviews**.
- A current **GET-only** Fleet Commander preflight passed on 12 observed target branches, with **zero writes**. Earlier in the same session, anonymous REST calls were temporarily blocked by the GitHub API's **60-request/hour unauthenticated quota**; rerun later succeeded after quota reset. A failed rate-limited scan is *not* evidence of a permission denial or protection state.
- The candidate policy is SHA-256 pinned as `461c6bd54985f4c1b0523788d3dd56216ff061b8f27a9c5136f561d781012efd`; baseline evidence is pinned as `dfc303135946a57064597b9097c80cdcbdb471959e6afd61d92bf45113a6a599`. **Neither file was changed in V14.**
- Five exact currently configured required contexts were observed, with GitHub Actions app ID **15368**: `lint-and-test (3.12)`, `focused-tests (3.12)`, `qualification-tests (3.12)`, `operational-failure-injection (3.12)`, `prompt-cache-tests (3.10)`.
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
3. After documenting all five decisions and verifying credential provenance, the human administrator may choose **one** manual GitHub UI activation or separately authorized single-create API action. No assistant/agent automatic activation. Avoid retrying a create request with uncertain outcome.
4. Independently inspect the active rule detail **and effective per-branch rules**. Use the GET-only `exp053_verify_live_enforcement.py`; metadata check must report `POLICY_METADATA_VERIFIED` with no unverified bypass. A named ruleset listing alone is insufficient.
5. An independent human uses the V9–V13 [negative merge-denial protocol](EXP053_V9_NEGATIVE_MERGE_CHALLENGE_HANDOFF.md): two otherwise mergeable **non-draft** harmless fixture PRs, one against `main` and one representative protected research branch, with passing required checks. Capture GitHub's *specific lack-of-human-approvals* refusal, without invoking Merge or enabling auto-merge. Capture current SHA, observed ruleset ID, human identity, screenshots/evidence digest in separately controlled storage. Draft PRs #37–#46 **cannot** serve as negative challenges.
6. Nominate and request actual human review on exact draft PR heads only when the independent review gate exists. Keep the draft PRs unmerged until genuine approvals and policy-readback evidence are recorded. Any merge must be a separate authorized maintainer decision.

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
