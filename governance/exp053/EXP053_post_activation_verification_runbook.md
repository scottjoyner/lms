# EXP-053 — administrator activation and independent read-back supplement

Date: 2026-10-20. Repository: `scottjoyner/lms`. Governance issue: [#47](https://github.com/scottjoyner/lms/issues/47). Science issue: [#31](https://github.com/scottjoyner/lms/issues/31).

**Current decision: `NO_GO_RESEARCH_ONLY`.** Neither a successful offline validator nor a future successful metadata read-back constitutes authorization for scientific inference, source corpus work, signer enrollment, production deployment, or merges.

## Delta since original handoff

- Recovered **the original four files** from ChatGPT Library without reconstructing their contents. Original candidate's offline preflight passes.
- Checked live public GitHub REST metadata on x1-370: `/repos/scottjoyner/lms/rulesets` returned `[]`; `/repos/scottjoyner/lms/rules/branches/main` returned `[]`. **No active effective rules were observable on `main`.** This read-only check performed no writes.
- Confirmed the connected GitHub app has repository `admin:true`, but it exposes **no repository-ruleset create/update action**. The `gh` CLI on x1-370 reports it is **unauthenticated**. Do not mistake GitHub permission metadata for a usable administrative credential in the shell.
- `/home/scott/git/lms` has substantial unrelated uncommitted edits. Do **not** use it for checkout/reset/clean to administer EXP-053. This entire supplement is standalone.
- Added a read-only `exp053_verify_live_enforcement.py` with **eight synthetic tests**. GitHub documents that effective branch rules are available from `/repos/{owner}/{repo}/rules/branches/{branch}` and exclude disabled/evaluate policies. The script checks the actual policy and each observed existing EXP-053 branch.

## Administrator action (do not run automatically)

1. Review the existing `EXP053_github_ruleset_candidate_2026-10-20.json`, the original runbook and issue #47. Confirm the exact observed CI check contexts and GitHub Actions application ID **15368** remain valid immediately before activation. Obtain explicit owner approval for the ruleset and a recovery process.
2. In the authenticated GitHub UI, visit [repository rules](https://github.com/scottjoyner/lms/settings/rules). Create/activate a **branch** ruleset covering `main` **and** `research/exp053*`, with no excludes or ordinary bypass actors. Require a PR, **2 non-author approvals**, stale review dismissal, last-push independent approval, resolved conversations, the **five exact GitHub Actions checks**, no force push and no deletion. Verify the target preview against every existing branch and future nested-branch naming conventions.
3. Take account of strict status checks across stacked PR branches; base updates can invalidate checks/reviews. Plan the review/merge sequence. Do not make PRs ready, request reviews from invented handles, or merge as part of activation.
4. After activation, independently read back. If using an authorized admin terminal, run `python3 EXP053_validate_ruleset_candidate_2026-10-20.py --dry-run`, then `python3 -m unittest -v test_exp053_verify_live_enforcement.py`, then `python3 exp053_verify_live_enforcement.py --json-out EXP053_live_readback.json`. The last script **only makes GET requests** and exits with status **2** for missing/mismatched/unknown enforcement. Set `GITHUB_TOKEN` in the operator's secure terminal environment *if needed*; never save a token to the report or repo. Anonymous public API responses **may redact bypass actors**; an unknown value therefore fails closed.
5. Review each line of `EXP053_live_readback.json`: `POLICY_METADATA_VERIFIED` means metadata checks passed, **not** that merge rejection was tested, and never changes `scientific_status=NO_GO_RESEARCH_ONLY`.
6. An independent operator should attempt a **non-destructive, unapproved merge-eligibility challenge** using an isolated PR against a target branch, without actually merging. Confirm missing reviews, invalid/absent required check, stale approval, latest push, unresolved thread and alternate/bypass pathway are blocked in GitHub's real UI/API. Capture links/screenshots and identity of the human verifier; do not fabricate attestations. Never force-push/deletion-test production branches. Record the challenge and rerun policy read-back before declaring governance enforcement operational.
7. Nominate two genuine independent source-review humans with verified GitHub handles and adequate permissions. Only after policy and actual reviewers are established can #37–#46 begin accountable review on their **current exact heads**. Source reviews and 40 independently adjudicated RACE items are separate requirements.

## Evidence limits and fail-closed behavior

- Public `GET /rulesets/{id}` can **omit** `bypass_actors` for an unauthorized reader. It must **not** be interpreted as `[]`; the verifier requires an explicit empty list for metadata verification.
- A reported `branch.protected=true` alone does **not** prove the required rules are in effect. The tool compares active effective rule parameters for **each** observed target branch, plus the named active ruleset.
- Branches named `research/exp053-nested/child` will **not** match the proposed `research/exp053*` pattern. The verifier flags this expansion; an administrator must adjust scope explicitly if such targets are legitimate.
- Future branch creation, changes to GitHub CI names/app identity, bypass settings, GitHub feature behavior, branch deletion and unsampled failure modes require fresh review. Automated evidence is complementary to, not a replacement for, real negative merge challenges and independent reviewer custody.
- Zero reviewer approvals, no operator key custody, absent off-host rollback-resistant high-water, and 40 independent human review items remain independent scientific blockers.

No ruleset, GitHub branch protection, source repository, PR, server, signed evidence, reviewer identity or deployment was modified by this package.

Official references: [Rules REST API](https://docs.github.com/en/rest/repos/rules) and [Ruleset behavior](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/available-rules-for-rulesets).
