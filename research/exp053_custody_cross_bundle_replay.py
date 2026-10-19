#!/usr/bin/env python3
"""Source-free, offline cross-bundle *negative* replay triage; NEVER approval.

Every signed receipt key is self-supplied, not a custodian trust root. This
checks only currently supplied bundles; absence of prior bundles is unknowable.
"""

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import exp053_custody_boundary_audit as custody
import exp053_custody_negative_challenges as challenges
import exp053_custody_receipt_consistency as receipts

SPEC = Path(__file__).resolve().parent / "EXP053-CUSTODY-CROSS-BUNDLE-REPLAY-SPEC-2026-10-18.md"
SPEC_SHA256 = "33774931fce461664335e3c3e6402c838266cf2d7ba543849f709f93d485d296"
SCHEMA = "exp053-custody-cross-bundle-replay-v1"
MAX_BUNDLES = 12


def audit(claim_bytes, bundles, *, now=None):
    """Require disjoint receipt scenarios across one untrusted signing context."""
    if (SPEC.is_symlink() or not SPEC.is_file() or
            hashlib.sha256(SPEC.read_bytes()).hexdigest() != SPEC_SHA256):
        raise custody.Hold("frozen cross-bundle replay spec changed")
    if not isinstance(bundles, (tuple, list)) or not 1 <= len(bundles) <= MAX_BUNDLES:
        raise custody.Hold("invalid bundle count")
    if now is None:
        now = datetime.now(timezone.utc)
    expected = challenges.plan(claim_bytes)
    scenario_ids = tuple(row["scenario_id"] for row in expected["scenario_rows"])
    known_scenarios = set()
    known_nonces = set()
    known_signatures = set()
    known_bundles = set()
    signer = None
    checkpoint = None
    empty = 0
    totals = Counter()
    for raw in bundles:
        # All original frozen signed-receipt validation, including local UTC
        # checks and Ed25519 source+plan binding, stays mandatory.
        receipts.assess(claim_bytes, raw, now=now)
        parsed = receipts._strict_load(raw)
        canonical = receipts._canonical(parsed)
        digest = hashlib.sha256(canonical).digest()
        if digest in known_bundles:
            raise custody.Hold("duplicate normalized receipt bundle")
        known_bundles.add(digest)
        rows = parsed["receipts"]
        if not rows:
            empty += 1
            continue
        if signer is None:
            signer = parsed["observer_public_key_b64"]
            checkpoint = parsed["checkpoint_reference_sha256"]
        elif (parsed["observer_public_key_b64"] != signer or
              parsed["checkpoint_reference_sha256"] != checkpoint):
            raise custody.Hold("changed observer identity or checkpoint context")
        for row in rows:
            if row["scenario_id"] in known_scenarios:
                raise custody.Hold("repeated scenario claim across bundles")
            if row["nonce_hex"] in known_nonces:
                raise custody.Hold("replayed nonce across bundles")
            if row["signature_hex"] in known_signatures:
                raise custody.Hold("replayed signature across bundles")
            known_scenarios.add(row["scenario_id"])
            known_nonces.add(row["nonce_hex"])
            known_signatures.add(row["signature_hex"])
            totals[row["outcome"]] += 1
    if empty and len(bundles) != 1:
        raise custody.Hold("empty baseline cannot mix with populated bundles")
    return {
        "schema": SCHEMA,
        "candidate_alias": expected["candidate_alias"],
        "bundle_count": len(bundles),
        "unique_scenario_claim_count": len(known_scenarios),
        "missing_scenario_ids": [scenario for scenario in scenario_ids
                                 if scenario not in known_scenarios],
        "claimed_denied_count": totals["DENIED"],
        "claimed_bypass_count": totals["BYPASS"],
        "claimed_unknown_count": totals["UNKNOWN"],
        "decision": "NO_GO_RESEARCH_ONLY",
        "scientific_execution_authorized": False,
        "independent_administration_approved": False,
        "witness_durability_approved": False,
        "all_challenges_independently_attested": False,
        "signer_authority": "UNTRUSTED_SELF_SUPPLIED_KEY",
        "durable_replay_protection": "NOT_PROVEN",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claims", type=Path, required=True,
                        help="Source-free frozen EXP053 custody claims")
    parser.add_argument("--bundles", nargs="+", type=Path, required=True,
                        help="1–12 source-free, untrusted signed receipt bundles")
    args = parser.parse_args(argv)
    try:
        if len(args.bundles) > MAX_BUNDLES:
            raise custody.Hold("bundle count exceeds limit")
        result = audit(custody._read_bounded_regular_file(args.claims),
                       [receipts._read(path) for path in args.bundles])
    except (custody.Hold, OSError, ValueError):
        print('{"decision":"NO_GO_RESEARCH_ONLY","scientific_execution_authorized":false,'
              '"independent_administration_approved":false,'
              '"witness_durability_approved":false,'
              '"all_challenges_independently_attested":false,'
              '"status":"INVALID_OR_UNVERIFIABLE_INPUT"}')
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
