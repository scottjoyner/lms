#!/usr/bin/env python3
"""Verify only local *synthetic/future* publication-chain consistency, NO-GO.

No independently governed observer root or high-water publisher is configured.
Caller-provided key and digest pins do NOT establish independent custody.
"""

import argparse
import base64
import binascii
import hashlib
import hmac
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import exp053_custody_boundary_audit as custody
import exp053_custody_negative_challenges as challenges
import exp053_custody_receipt_consistency as receipts
import exp053_receipt_checkpoint_comparison as reference_compare

SPEC = Path(__file__).resolve().parent / "EXP053-OBSERVER-PUBLICATION-CHAIN-SPEC-2026-10-18.md"
SPEC_SHA256 = "a387aa42f93b55a8ad1325dbddd5621e906bfc165b23967e42f65e9ebceff5e5"
SCHEMA = "exp053-custody-observer-publication-chain-v1"
SIGNED_SCHEMA = "exp053-custody-observer-signed-checkpoint-v1"
REPORT_SCHEMA = "exp053-observer-publication-chain-audit-v1"
DOMAIN = b"EXP053-OBSERVER-PUBLICATION-V1\x00"
FIELDS = {"schema", "candidate_alias", "plan_sha256", "checkpoints"}
ROW_FIELDS = {"sequence", "previous_statement_sha256", "reference_sha256",
              "observed_at_utc", "signature_hex"}
MAX_PUBLICATION_BYTES = 32768
ZERO = "0" * 64


def _frozen():
    if (SPEC.is_symlink() or not SPEC.is_file() or
            hashlib.sha256(SPEC.read_bytes()).hexdigest() != SPEC_SHA256):
        raise custody.Hold("frozen publication-chain specification changed")


def _observer_key(public_b64, expected_digest):
    if not isinstance(public_b64, str) or not receipts._sha(expected_digest):
        raise custody.Hold("missing claimed external observer key and digest")
    try:
        binary = base64.b64decode(public_b64, validate=True)
        if len(binary) != 32 or base64.b64encode(binary).decode() != public_b64:
            raise custody.Hold("noncanonical observer public key")
        if not hmac.compare_digest(hashlib.sha256(binary).hexdigest(), expected_digest):
            raise custody.Hold("observer key differs from supplied root")
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        return Ed25519PublicKey.from_public_bytes(binary)
    except (ImportError, binascii.Error, ValueError) as exc:
        raise custody.Hold("unusable claimed observer key") from exc


def _decode_publication(raw):
    if not isinstance(raw, bytes) or not 1 <= len(raw) <= MAX_PUBLICATION_BYTES:
        raise custody.Hold("publication outside byte limit")
    obj = receipts._strict_load(raw)
    if raw != receipts._canonical(obj) + b"\n":
        raise custody.Hold("noncanonical publication bytes")
    if (type(obj) is not dict or set(obj) != FIELDS or obj["schema"] != SCHEMA or
            type(obj["checkpoints"]) is not list or not 1 <= len(obj["checkpoints"]) <= 13):
        raise custody.Hold("invalid publication structure")
    return obj


def verify(claim_bytes, bundle_bytes, reference_bytes, reference_digest,
           publication_bytes, observer_public_key_b64, expected_observer_key_sha256,
           expected_head_sha256, *, now=None):
    """Check an untrusted chain bound to the already frozen receipt snapshot.

    The caller can replace every key/hash argument. A locally consistent chain
    never proves that another administrator retained the expected head.
    """
    _frozen()
    if not receipts._sha(expected_head_sha256):
        raise custody.Hold("missing claimed publication head digest")
    verifier = _observer_key(observer_public_key_b64, expected_observer_key_sha256)
    if now is None:
        now = datetime.now(timezone.utc)
    if (not isinstance(now, datetime) or now.tzinfo is None or
            now.utcoffset() != timedelta(0)):
        raise custody.Hold("unavailable local UTC verifier clock")
    # Uses all previous frozen receipt, collection and reference checks.
    reference_compare.compare(claim_bytes, bundle_bytes, reference_bytes,
                              reference_digest, now=now)
    expected = challenges.plan(claim_bytes)
    plan_sha = hashlib.sha256(receipts._canonical(expected)).hexdigest()
    reference = receipts._strict_load(reference_bytes)
    publication = _decode_publication(publication_bytes)
    if (publication["candidate_alias"] != expected["candidate_alias"] or
            publication["plan_sha256"] != plan_sha):
        raise custody.Hold("publication candidate or plan differs")
    head = ZERO
    latest_time = None
    latest_reference = None
    for i, row in enumerate(publication["checkpoints"]):
        if (type(row) is not dict or set(row) != ROW_FIELDS or
                type(row["sequence"]) is not int or row["sequence"] != i or
                not receipts._sha(row["previous_statement_sha256"]) or
                row["previous_statement_sha256"] != head or
                not receipts._sha(row["reference_sha256"]) or
                not isinstance(row["signature_hex"], str) or
                not receipts.HEX128.fullmatch(row["signature_hex"])):
            raise custody.Hold("invalid sequence, previous head or signature shape")
        timestamp = receipts._utc(row["observed_at_utc"])
        if timestamp > now + timedelta(seconds=300) or (
                latest_time is not None and timestamp < latest_time):
            raise custody.Hold("invalid local-clock checkpoint chronology")
        unsigned = {
            "schema": SIGNED_SCHEMA,
            "candidate_alias": expected["candidate_alias"],
            "plan_sha256": plan_sha,
            **{field: row[field] for field in ROW_FIELDS if field != "signature_hex"},
        }
        try:
            verifier.verify(bytes.fromhex(row["signature_hex"]),
                            DOMAIN + receipts._canonical(unsigned))
        except Exception as exc:
            raise custody.Hold("untrusted observer signature invalid") from exc
        signed = {**unsigned, "signature_hex": row["signature_hex"]}
        head = hashlib.sha256(receipts._canonical(signed)).hexdigest()
        latest_time = timestamp
        latest_reference = row["reference_sha256"]
    if (reference["checkpoint_sequence"] != len(publication["checkpoints"]) - 1 or
            not hmac.compare_digest(latest_reference, reference_digest) or
            not hmac.compare_digest(head, expected_head_sha256)):
        raise custody.Hold("published head or checkpoint reference mismatch")
    return {
        "schema": REPORT_SCHEMA,
        "candidate_alias": expected["candidate_alias"],
        "published_statement_count": len(publication["checkpoints"]),
        "observed_head_sequence": len(publication["checkpoints"]) - 1,
        "head_reference_matches_local": True,
        "observer_public_key_matches_caller_pin": True,
        "head_matches_caller_pin": True,
        "publication_custody": "CALLER_SUPPLIED_UNTRUSTED",
        "durable_replay_protection": "NOT_PROVEN",
        "scientific_execution_authorized": False,
        "independent_administration_approved": False,
        "witness_durability_approved": False,
        "decision": "NO_GO_RESEARCH_ONLY",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claims", required=True, type=Path)
    parser.add_argument("--bundles", required=True, type=Path, nargs="+")
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--expected-reference-sha256", required=True)
    parser.add_argument("--publication", required=True, type=Path)
    parser.add_argument("--observer-public-key-b64", required=True)
    parser.add_argument("--expected-observer-key-sha256", required=True)
    parser.add_argument("--expected-head-sha256", required=True)
    args = parser.parse_args(argv)
    try:
        if len(args.bundles) > 12:
            raise custody.Hold("excessive source-free receipt bundles")
        result = verify(
            custody._read_bounded_regular_file(args.claims),
            [receipts._read(path) for path in args.bundles],
            custody._read_bounded_regular_file(args.reference),
            args.expected_reference_sha256,
            receipts._read(args.publication),
            args.observer_public_key_b64, args.expected_observer_key_sha256,
            args.expected_head_sha256)
    except (custody.Hold, ValueError, OSError):
        print('{"decision":"NO_GO_RESEARCH_ONLY","scientific_execution_authorized":false,'
              '"independent_administration_approved":false,'
              '"witness_durability_approved":false,'
              '"status":"INVALID_OR_UNTRUSTED_PUBLICATION"}')
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
