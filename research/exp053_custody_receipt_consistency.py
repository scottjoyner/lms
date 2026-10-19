#!/usr/bin/env python3
"""Fail-closed source-free receipt consistency screen; NEVER custody approval.

Only checks a self-supplied observer's signature over synthetic/future claims.
Signature validity is not independent administration, rollback proof or trust.
"""

import argparse
import base64
import binascii
import hashlib
import json
import os
import re
import stat
from datetime import datetime, timedelta, timezone
from pathlib import Path

import exp053_custody_boundary_audit as custody
import exp053_custody_negative_challenges as challenges

SPEC = Path(__file__).resolve().parent / "EXP053-CUSTODY-RECEIPT-CONSISTENCY-SPEC-2026-10-18.md"
SPEC_SHA256 = "73fa530a2704a857b449931150d48e63b7b15f3d07be0ca0055f2ba23268e36a"
BUNDLE_SCHEMA = "exp053-custody-negative-receipt-bundle-v1"
RECEIPT_SCHEMA = "exp053-custody-negative-receipt-v1"
REPORT_SCHEMA = "exp053-custody-receipt-consistency-v1"
MAX_BYTES = 48 * 1024
DOMAIN = b"EXP053-CUSTODY-NEGATIVE-RECEIPT-V1\x00"
BUNDLE_FIELDS = {
    "schema", "plan_sha256", "candidate_alias", "observer_public_key_b64",
    "checkpoint_reference_sha256", "receipts",
}
ROW_FIELDS = {
    "scenario_id", "outcome", "nonce_hex", "evidence_sha256",
    "observed_at_utc", "signature_hex",
}
UTC_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z\Z")
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
HEX128 = re.compile(r"[0-9a-f]{128}\Z")
OUTCOMES = {"DENIED", "BYPASS", "UNKNOWN"}


def _sha(value):
    return isinstance(value, str) and bool(HEX64.fullmatch(value))


def _frozen():
    if (SPEC.is_symlink() or not SPEC.is_file() or
            hashlib.sha256(SPEC.read_bytes()).hexdigest() != SPEC_SHA256):
        raise custody.Hold("frozen receipt consistency specification changed")


def _canonical(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    except (RecursionError, ValueError, TypeError, OverflowError) as exc:
        # Canonicalization is also applied before some callers validate the
        # exact source-free shape. Untrusted nested JSON must fail closed.
        raise custody.Hold("unusable or deeply nested canonical JSON") from exc


def _read(path):
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        fd = os.open(path, flags)
        with os.fdopen(fd, "rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise custody.Hold("input is not a regular file")
            return stream.read(MAX_BYTES + 1)
    except OSError as exc:
        raise custody.Hold("cannot safely open receipt bundle") from exc


def _strict_load(raw):
    if not isinstance(raw, bytes) or not 1 <= len(raw) <= MAX_BYTES:
        raise custody.Hold("receipt bundle outside byte limit")
    try:
        return json.loads(raw.decode("utf-8", errors="strict"),
                          object_pairs_hook=custody._duplicates,
                          parse_constant=custody._reject_nonfinite)
    except (ValueError, TypeError, UnicodeError, RecursionError) as exc:
        raise custody.Hold("malformed, ambiguous or deeply nested receipt bundle") from exc


def _utc(value):
    if not isinstance(value, str) or not UTC_RE.fullmatch(value):
        raise custody.Hold("invalid receipt UTC encoding")
    try:
        moment = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise custody.Hold("impossible receipt UTC time") from exc
    if moment.strftime("%Y-%m-%dT%H:%M:%SZ") != value:
        raise custody.Hold("noncanonical receipt UTC time")
    return moment


def assess(claim_bytes, receipt_bytes, *, now=None):
    _frozen()
    # The frozen original custody + challenge contracts are checked by plan().
    expected = challenges.plan(claim_bytes)
    expected_sha = hashlib.sha256(_canonical(expected)).hexdigest()
    valid_ids = tuple(row["scenario_id"] for row in expected["scenario_rows"])
    bundle = _strict_load(receipt_bytes)
    if (type(bundle) is not dict or set(bundle) != BUNDLE_FIELDS or
            bundle["schema"] != BUNDLE_SCHEMA or
            bundle["plan_sha256"] != expected_sha or
            bundle["candidate_alias"] != expected["candidate_alias"] or
            type(bundle["receipts"]) is not list or
            len(bundle["receipts"]) > len(valid_ids)):
        raise custody.Hold("receipt bundle does not match frozen plan")
    rows = bundle["receipts"]
    key = bundle["observer_public_key_b64"]
    checkpoint = bundle["checkpoint_reference_sha256"]
    if not rows:
        if key is not None or checkpoint is not None:
            raise custody.Hold("empty receipts cannot assert signer or checkpoint")
    else:
        if not isinstance(key, str) or not _sha(checkpoint):
            raise custody.Hold("missing claimed signer or checkpoint")
        try:
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
            binary = base64.b64decode(key, validate=True)
            if len(binary) != 32 or base64.b64encode(binary).decode() != key:
                raise custody.Hold("uncanonical observer key")
            verifier = Ed25519PublicKey.from_public_bytes(binary)
        except (ImportError, binascii.Error, ValueError) as exc:
            raise custody.Hold("observer verifier unavailable or invalid") from exc
    if now is None:
        now = datetime.now(timezone.utc)
    if (not isinstance(now, datetime) or now.tzinfo is None or
            now.utcoffset() != timedelta(0)):
        raise custody.Hold("local verifier clock unavailable")
    found = set()
    nonces = set()
    tally = {name: 0 for name in OUTCOMES}
    for row in rows:
        if type(row) is not dict or set(row) != ROW_FIELDS:
            raise custody.Hold("incorrect receipt row shape")
        scenario = row["scenario_id"]
        if not isinstance(scenario, str) or scenario not in valid_ids or scenario in found:
            raise custody.Hold("unknown or repeated scenario")
        found.add(scenario)
        outcome = row["outcome"]
        if not isinstance(outcome, str) or outcome not in OUTCOMES:
            raise custody.Hold("invalid self-reported outcome")
        if not _sha(row["nonce_hex"]) or row["nonce_hex"] in nonces:
            raise custody.Hold("invalid or repeated challenge nonce")
        nonces.add(row["nonce_hex"])
        if not _sha(row["evidence_sha256"]):
            raise custody.Hold("invalid claimed evidence digest")
        observed = _utc(row["observed_at_utc"])
        if now - observed > timedelta(hours=24) or observed - now > timedelta(seconds=300):
            raise custody.Hold("stale or future self-reported observation")
        if not isinstance(row["signature_hex"], str) or not HEX128.fullmatch(row["signature_hex"]):
            raise custody.Hold("incorrect Ed25519 signature encoding")
        unsigned = {
            "schema": RECEIPT_SCHEMA,
            "plan_sha256": expected_sha,
            "candidate_alias": expected["candidate_alias"],
            "checkpoint_reference_sha256": checkpoint,
            **{field: row[field] for field in ROW_FIELDS if field != "signature_hex"},
        }
        try:
            verifier.verify(bytes.fromhex(row["signature_hex"]), DOMAIN + _canonical(unsigned))
        except Exception as exc:
            raise custody.Hold("untrusted observer signature invalid") from exc
        tally[outcome] += 1
    return {
        "schema": REPORT_SCHEMA,
        "candidate_alias": expected["candidate_alias"],
        "decision": "NO_GO_RESEARCH_ONLY",
        "scientific_execution_authorized": False,
        "independent_administration_approved": False,
        "witness_durability_approved": False,
        "all_challenges_independently_attested": False,
        "signer_authority": "UNTRUSTED_SELF_SUPPLIED_KEY",
        "expected_scenario_count": len(valid_ids),
        "cryptographically_consistent_untrusted_receipt_count": len(found),
        "missing_scenario_ids": [item for item in valid_ids if item not in found],
        "claimed_denied_count": tally["DENIED"],
        "claimed_bypass_count": tally["BYPASS"],
        "claimed_unknown_count": tally["UNKNOWN"],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claims", type=Path, required=True,
                        help="Source-free frozen custody claims JSON")
    parser.add_argument("--receipts", type=Path, required=True,
                        help="Source-free untrusted receipt bundle JSON")
    args = parser.parse_args(argv)
    try:
        report = assess(custody._read_bounded_regular_file(args.claims), _read(args.receipts))
    except (custody.Hold, OSError, ValueError):
        print('{"decision":"NO_GO_RESEARCH_ONLY","scientific_execution_authorized":false,'
              '"independent_administration_approved":false,'
              '"witness_durability_approved":false,'
              '"all_challenges_independently_attested":false,'
              '"status":"INVALID_OR_UNVERIFIABLE_INPUT"}')
        return 2
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
