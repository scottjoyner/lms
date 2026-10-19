#!/usr/bin/env python3
"""Read-only past-reference consistency screen; never anti-rollback authority.

An operator supplied both a checkpoint and its alleged pin. This tool does not
know whether either was ever independently published or protected from rollback.
"""

import argparse
import hashlib
import hmac
import json
from datetime import datetime, timezone
from pathlib import Path

import exp053_custody_boundary_audit as custody
import exp053_custody_cross_bundle_replay as collection
import exp053_custody_negative_challenges as challenges
import exp053_custody_receipt_consistency as receipts

SPEC = Path(__file__).resolve().parent / "EXP053-RECEIPT-CHECKPOINT-COMPARISON-SPEC-2026-10-18.md"
SPEC_SHA256 = "aa7d4cd22df8a8b9a094c0172789398560672e38271b5fb60e9e11918d22f914"
REFERENCE_SCHEMA = "exp053-receipt-checkpoint-reference-v1"
REPORT_SCHEMA = "exp053-receipt-checkpoint-comparison-v1"
REFERENCE_FIELDS = {"schema", "candidate_alias", "plan_sha256", "checkpoint_sequence", "receipt_entries"}
ENTRY_FIELDS = {"scenario_id", "receipt_sha256"}


def _frozen():
    if (SPEC.is_symlink() or not SPEC.is_file() or
            hashlib.sha256(SPEC.read_bytes()).hexdigest() != SPEC_SHA256):
        raise custody.Hold("frozen checkpoint comparison spec changed")


def compare(claim_bytes, bundle_bytes, reference_bytes, expected_reference_sha256, *, now=None):
    """Reject omitted/substituted prior rows from the presented collection.

    Comparing snapshots is NOT historical rollback protection. If a hostile
    scorer can change both the snapshot and its supplied hash, it can deceive
    this comparison without ever changing a real external witness.
    """
    _frozen()
    if not receipts._sha(expected_reference_sha256):
        raise custody.Hold("missing externally specified reference digest")
    # Reject an input digest typo before processing an untrusted reference.
    if (not isinstance(reference_bytes, bytes) or
            not 1 <= len(reference_bytes) <= custody.MAX_BYTES or
            not hmac.compare_digest(hashlib.sha256(reference_bytes).hexdigest(),
                                    expected_reference_sha256)):
        raise custody.Hold("reference bytes differ from caller pin")
    reference = receipts._strict_load(reference_bytes)
    if reference_bytes != receipts._canonical(reference) + b"\n":
        raise custody.Hold("noncanonical reference encoding")
    expected = challenges.plan(claim_bytes)
    plan_digest = hashlib.sha256(receipts._canonical(expected)).hexdigest()
    scenario_ids = tuple(row["scenario_id"] for row in expected["scenario_rows"])
    if (type(reference) is not dict or set(reference) != REFERENCE_FIELDS or
            reference["schema"] != REFERENCE_SCHEMA or
            reference["candidate_alias"] != expected["candidate_alias"] or
            reference["plan_sha256"] != plan_digest or
            type(reference["checkpoint_sequence"]) is not int or
            not 0 <= reference["checkpoint_sequence"] <= len(scenario_ids) or
            type(reference["receipt_entries"]) is not list or
            len(reference["receipt_entries"]) != reference["checkpoint_sequence"]):
        raise custody.Hold("reference schema, plan or count mismatch")
    older = {}
    for item in reference["receipt_entries"]:
        if (type(item) is not dict or set(item) != ENTRY_FIELDS or
                not isinstance(item["scenario_id"], str) or
                item["scenario_id"] not in scenario_ids or
                not receipts._sha(item["receipt_sha256"]) or
                item["scenario_id"] in older):
            raise custody.Hold("malformed or duplicate referenced scenario")
        older[item["scenario_id"]] = item["receipt_sha256"]
    if list(older) != [key for key in scenario_ids if key in older]:
        raise custody.Hold("reference scenario order not canonical")
    if now is None:
        now = datetime.now(timezone.utc)
    # This re-validates every signed receipt, collection nonce uniqueness and
    # same self-supplied observer key using the original frozen protocols.
    collection.audit(claim_bytes, bundle_bytes, now=now)
    fresh = {}
    for bundle in bundle_bytes:
        parsed = receipts._strict_load(bundle)
        for row in parsed["receipts"]:
            scenario = row["scenario_id"]
            if scenario in fresh:
                raise custody.Hold("repeated current scenario")
            fresh[scenario] = hashlib.sha256(receipts._canonical(row)).hexdigest()
    for scenario, old_digest in older.items():
        if scenario not in fresh:
            raise custody.Hold("previously referenced receipt omitted")
        if not hmac.compare_digest(old_digest, fresh[scenario]):
            raise custody.Hold("previously referenced signed receipt changed")
    return {
        "schema": REPORT_SCHEMA,
        "candidate_alias": expected["candidate_alias"],
        "prior_receipt_count": len(older),
        "current_untrusted_receipt_count": len(fresh),
        "additional_untrusted_receipt_count": len(fresh) - len(older),
        "historical_receipts_consistent": True,
        "reference_custody": "CALLER_SUPPLIED_UNTRUSTED",
        "durable_replay_protection": "NOT_PROVEN",
        "decision": "NO_GO_RESEARCH_ONLY",
        "independent_administration_approved": False,
        "witness_durability_approved": False,
        "scientific_execution_authorized": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claims", required=True, type=Path,
                        help="source-free frozen custody manifest")
    parser.add_argument("--bundles", required=True, nargs="+", type=Path,
                        help="1–12 untrusted source-free receipt bundles")
    parser.add_argument("--reference", required=True, type=Path,
                        help="previously captured source-free reference snapshot")
    parser.add_argument("--expected-reference-sha256", required=True,
                        help="exact reference-file digest separately supplied by caller (untrusted)")
    args = parser.parse_args(argv)
    try:
        if len(args.bundles) > collection.MAX_BUNDLES:
            raise custody.Hold("too many bundles")
        result = compare(
            custody._read_bounded_regular_file(args.claims),
            [receipts._read(item) for item in args.bundles],
            custody._read_bounded_regular_file(args.reference),
            args.expected_reference_sha256)
    except (custody.Hold, ValueError, OSError):
        print('{"decision":"NO_GO_RESEARCH_ONLY","scientific_execution_authorized":false,'
              '"independent_administration_approved":false,'
              '"witness_durability_approved":false,'
              '"status":"INVALID_OR_INCONSISTENT_UNTRUSTED_REFERENCE"}')
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
