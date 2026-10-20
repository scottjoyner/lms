#!/usr/bin/env python3
"""Read-only local preflight for EXP-053 GitHub repository ruleset candidate.

Does NOT contact GitHub and cannot create, edit or enable rulesets.
It checks a snapshot against the proposed payload. After installation,
verify active GitHub policy with an authorized repository administrator.
"""
import argparse
import fnmatch
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
RULE = BASE / "EXP053_github_ruleset_candidate_2026-10-20.json"
EVIDENCE = BASE / "EXP053_ruleset_validation_evidence_2026-10-20.json"


def checked(condition, reason):
    if not condition:
        raise AssertionError(reason)


def matches_ref(pattern, ref):
    """Github's ruleset fnmatch behavior does not let '*' consume '/'."""
    p = pattern.split("/")
    r = ref.split("/")
    return len(p) == len(r) and all(fnmatch.fnmatchcase(b, a) for a, b in zip(p, r))


def applies(refs, ref):
    return any(matches_ref(x, ref) for x in refs.get("include", [])) and not any(
        matches_ref(x, ref) for x in refs.get("exclude", [])
    )


def validate():
    ruleset = json.loads(RULE.read_text())
    evidence = json.loads(EVIDENCE.read_text())
    checked(ruleset["enforcement"] == "active", "Not active: would not enforce")
    checked(ruleset["target"] == "branch", "Not scoped to branch rules")
    checked(ruleset["bypass_actors"] == [], "Bypass actors must be owner-reviewed")
    ref = ruleset["conditions"]["ref_name"]
    checked(ref["include"] == ["refs/heads/main", "refs/heads/research/exp053*"],
            "Unexpected branch scope")
    checked(ref["exclude"] == [], "Unexpected exclusions")
    names = ["main"] + evidence["baseline"]["research_exp053_branches"]
    checked(len(names) == len(set(names)) == 12, "Expected one default plus eleven exp053 branches")
    for name in names:
        checked(applies(ref, "refs/heads/" + name), "Unprotected desired branch " + name)
    # Negative checks: avoid swallowing all 'research/*' and root branches.
    for name in ("feature/unrelated", "research/other", "dev", "research/exp054-scope"):
        checked(not applies(ref, "refs/heads/" + name), "Unexpected target " + name)
    # An additional slash must NOT be covered by a '*' glob.
    checked(not matches_ref(ref["include"][1], "refs/heads/research/exp053-deeper/nested"),
            "Wildcard incorrectly crosses a directory separator")

    types = {r["type"]: r for r in ruleset["rules"]}
    checked(set(types) == {"pull_request", "required_status_checks", "non_fast_forward", "deletion"},
            "Unexpected or missing rule type")
    pr = types["pull_request"]["parameters"]
    checked(pr["required_approving_review_count"] >= 2, "Too few independent approvals")
    for key in ("dismiss_stale_reviews_on_push", "require_last_push_approval",
                "required_review_thread_resolution"):
        checked(pr[key] is True, "Review protection disabled: " + key)
    c = types["required_status_checks"]["parameters"]
    checked(c["strict_required_status_checks_policy"] is True, "Non-strict check policy")
    contexts = c["required_status_checks"]
    checked(len(contexts) >= 5, "Not enough verified CI workflows")
    expected = evidence["verified_successful_check_contexts"]
    checked([z["context"] for z in contexts] == expected, "CI names diverge from observed checks")
    checked(all(z["integration_id"] == evidence["check_source"]["app_id"] == 15368
                for z in contexts), "Unexpected CI app source")
    checked(len(evidence["check_contexts_verified_at"]) == 11,
            "CI not checked on default branch plus all ten PR heads")
    checked(evidence["baseline"]["submitted_reviews_on_draft_prs"] == 0,
            "Snapshot unexpectedly claims reviewer submission")
    checked(evidence["baseline"]["scientific_authorization"] == "NO_GO_RESEARCH_ONLY",
            "Scientific hold absent")
    print("PASS: branch matching covers main and all 11 observed EXP-053 branches")
    print("PASS: unscoped branches are excluded, wildcard does not cross slash")
    print("PASS: >=2 human approvals, stale review dismissal, latest-push approval")
    print("PASS: no bypass actors, no force push, no deletion")
    print("PASS: 5 previously observed checks use GitHub Actions integration ID 15368")
    print("PASS: candidate contains only a proposal; no live ruleset was changed")
    print("CURRENT STATUS: NO_GO_RESEARCH_ONLY; ruleset requires admin activation")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Only supported mode; makes no changes")
    parser.parse_args()
    validate()
