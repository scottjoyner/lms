#!/usr/bin/env python3
"""Create an unexecuted, source-free EXP-053 independent-custody test checklist.

ALWAYS NO-GO. Not a witness, permission checker, or test executor.
"""

import argparse
import hashlib
import json
from pathlib import Path

import exp053_custody_boundary_audit as custody

SPEC = Path(__file__).resolve().parent / "EXP053-INDEPENDENT-CUSTODY-NEGATIVE-CHALLENGES-SPEC-2026-10-18.md"
SPEC_SHA256 = "d3308a8b627c513e5b8aff200da2befaaafc625e33a992ca7ad32a095e007744"
SCHEMA = "exp053-custody-negative-challenge-plan-v1"

# Fixed prospective *privilege probes*. Deliberately omit commands, secrets,
# network targets and executable steps. Custodian must design approved tests.
SCENARIOS = (
    ("SCORER_SSH_ACCESS", "witness_remote_login", "scorer_operator", "isolate witness login"),
    ("FLEET_SSH_ACCESS", "witness_remote_login", "fleet_automation", "isolate witness login"),
    ("SCORER_ROOT_CONTROL", "witness_root_administration", "scorer_operator", "deny root sudo console"),
    ("FLEET_ROOT_CONTROL", "witness_root_administration", "fleet_automation", "deny root sudo console"),
    ("SIGNER_READ_OR_USE", "witness_signing_key_custody", "scorer_operator", "deny signer key read use"),
    ("HIGHWATER_REWIND", "witness_highwater_mutation", "scorer_operator", "deny CAS rewind replay"),
    ("SNAPSHOT_RESTORE", "witness_snapshot_restore", "scorer_operator", "deny snapshot rollback"),
    ("BACKUP_RESTORE", "witness_backup_restore", "fleet_automation", "deny backup rollback"),
    ("TRUST_ROOT_REPLACEMENT", "witness_trust_anchor_rotation", "scorer_operator", "deny trust root replacement"),
    ("CHECKPOINT_FORK", "witness_checkpoint_publication", "scorer_operator", "deny checkpoint fork"),
    ("FLEET_SIGNER_ACCESS", "witness_signing_key_custody", "fleet_automation", "deny signer key read use"),
    ("FLEET_CAS_ACCESS", "witness_highwater_mutation", "fleet_automation", "deny CAS state mutation"),
)


def plan(raw):
    """Never translate self-reported claims into evidence of executed tests."""
    if (SPEC.is_symlink() or not SPEC.is_file() or
            hashlib.sha256(SPEC.read_bytes()).hexdigest() != SPEC_SHA256):
        raise custody.Hold("frozen challenge plan specification changed")
    manifest = custody.parse(raw)
    lookup = {(item["domain"], item["principal"]): item for item in manifest["claims"]}
    rows = []
    for scenario_id, domain, principal, challenge in SCENARIOS:
        assertion = lookup.get((domain, principal))
        if assertion is None:
            status = "MISSING"
        elif assertion["access"] == "CAN":
            status = "CLAIMED_CAN"
        elif assertion["access"] == "CANNOT":
            status = "CLAIMED_CANNOT_UNVERIFIED"
        else:
            status = "CLAIMED_UNKNOWN"
        rows.append({
            "scenario_id": scenario_id,
            "domain": domain,
            "actor": principal,
            "challenge": challenge,
            "expected": "DENIED_AND_INDEPENDENTLY_VERIFIED",
            "execution_status": "NOT_EXECUTED",
            "evidence_status": "NOT_ATTESTED",
            "claim_status": status,
        })
    return {
        "schema": SCHEMA,
        "candidate_alias": manifest["candidate_alias"],
        "input_sha256": hashlib.sha256(raw).hexdigest(),
        "decision": "NO_GO_RESEARCH_ONLY",
        "scientific_execution_authorized": False,
        "independent_administration_approved": False,
        "witness_durability_approved": False,
        "all_scenarios_executed": False,
        "scenario_count": len(rows),
        "scenario_rows": rows,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="Bounded local source-free v1 custody claims JSON")
    args = parser.parse_args(argv)
    try:
        # Use the same bounded, O_NOFOLLOW, same-descriptor read as the custody
        # negative screen. This does not invoke SSH or process private sources.
        result = plan(custody._read_bounded_regular_file(args.input))
    except (custody.Hold, OSError, ValueError):
        print('{"decision":"NO_GO_RESEARCH_ONLY","scientific_execution_authorized":false,'
              '"all_scenarios_executed":false,"status":"INVALID_OR_UNVERIFIABLE_INPUT"}')
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
