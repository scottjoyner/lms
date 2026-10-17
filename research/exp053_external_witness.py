#!/usr/bin/env python3
"""Fail-closed external high-water CAS receipt verifier for future EXP-053 runs.

The externally administered CAS backend DOES NOT EXIST in this implementation.
No real trust root, key or adapter is configured; tests can inject a strictly
synthetic backend to verify refusal behavior. No model inference happens here.
"""
import base64
import hashlib
import os
from pathlib import Path

from exp053_execution_attestation import hex_digest
from exp053_race_review import canonical, deny, filehash

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "research" / "EXP053-EXTERNAL-HIGHWATER-WITNESS-SPEC-2026-10-17.md"
SPEC_SHA256 = "3aa0547ee07a9c271fd67b6542544a147790fef052337f1ce1764a8c2cbb5f1e"
REQUEST_SCHEMA = "exp053-external-highwater-cas-request-v1"
RESPONSE_SCHEMA = "exp053-external-highwater-cas-response-v1"
REQUEST_FIELDS = {
    "schema", "nonce_hex", "sequence", "expected_previous_sequence",
    "previous_claim_sha256", "claim_sha256", "runner_execution_id_sha256",
}
RESPONSE_FIELDS = {
    "schema", "nonce_hex", "sequence", "previous_claim_sha256",
    "claim_sha256", "runner_execution_id_sha256", "witness_counter",
    "status", "witness_public_key_b64", "signature_hex",
}

# Real source requires independent operator source review, key custody and
# a genuine remote atomic CAS implementation. These are deliberately EMPTY.
APPROVED_EXTERNAL_WITNESS_PUBLIC_KEY_SHA256 = ""
APPROVED_EXTERNAL_WITNESS_ADAPTER_SHA256 = ""


def _deny(reason):
    deny("EXP053 EXTERNAL HIGH-WATER HOLD: " + reason)


def _external_cas(_request):
    """Must be replaced by an independently reviewed, real external CAS.

    Do not route to the same node, directory, or process that holds the
    operator or scorer journals. Never synthesize a success in production.
    """
    _deny("no independently hosted, approved high-water CAS backend installed")


def commit_independent_highwater(*, sequence, previous_claim_sha256,
                                 claim_sha256, execution_id_sha256,
                                 operator_public_key_sha256,
                                 scorer_public_key_sha256):
    """Require a nonce-fresh signed remote CAS commit BEFORE local mutations."""
    if not SPEC.is_file() or filehash(SPEC) != SPEC_SHA256:
        _deny("prospectively frozen high-water protocol changed")
    if (not hex_digest(APPROVED_EXTERNAL_WITNESS_PUBLIC_KEY_SHA256) or
            not hex_digest(APPROVED_EXTERNAL_WITNESS_ADAPTER_SHA256)):
        _deny("external witness authority or implementation is not approved")
    if (not hex_digest(operator_public_key_sha256) or
            not hex_digest(scorer_public_key_sha256) or
            len({APPROVED_EXTERNAL_WITNESS_PUBLIC_KEY_SHA256,
                 operator_public_key_sha256,
                 scorer_public_key_sha256}) != 3):
        _deny("external witness, operator, scorer must have independent keys")
    if (type(sequence) is not int or sequence < 1 or
            not all(hex_digest(d) for d in (
                previous_claim_sha256, claim_sha256, execution_id_sha256))):
        _deny("incomplete request high-water hashes or sequence")
    request = {
        "schema": REQUEST_SCHEMA,
        "nonce_hex": os.urandom(32).hex(),
        "sequence": sequence,
        "expected_previous_sequence": sequence - 1,
        "previous_claim_sha256": previous_claim_sha256,
        "claim_sha256": claim_sha256,
        "runner_execution_id_sha256": execution_id_sha256,
    }
    if set(request) != REQUEST_FIELDS:
        _deny("internal request schema mismatch")
    try:
        response = _external_cas(request)
    except Exception as exc:
        _deny("unavailable external CAS or conflicting high-water: " +
              type(exc).__name__)
    if not isinstance(response, dict) or set(response) != RESPONSE_FIELDS:
        _deny("unexpected external high-water response fields")
    if (response["schema"] != RESPONSE_SCHEMA or
            response["status"] != "committed" or
            type(response["witness_counter"]) is not int or
            response["witness_counter"] != sequence or
            type(response["sequence"]) is not int or
            response["sequence"] != sequence):
        _deny("witness did not attest a committed matching sequence")
    for field in ("nonce_hex", "previous_claim_sha256", "claim_sha256",
                  "runner_execution_id_sha256"):
        if not isinstance(response[field], str) or response[field] != request[field]:
            _deny("witness response stale, replayed, or bound to wrong " + field)
    try:
        pub = base64.b64decode(response["witness_public_key_b64"], validate=True)
        if (len(pub) != 32 or
                base64.b64encode(pub).decode() != response["witness_public_key_b64"] or
                hashlib.sha256(pub).hexdigest() !=
                APPROVED_EXTERNAL_WITNESS_PUBLIC_KEY_SHA256):
            _deny("untrusted or noncanonical independent witness public key")
        signature = bytes.fromhex(response["signature_hex"])
        if len(signature) != 64:
            _deny("wrong external witness signature length")
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        unsigned = {key: value for key, value in response.items()
                    if key != "signature_hex"}
        Ed25519PublicKey.from_public_bytes(pub).verify(signature, canonical(unsigned))
    except Exception as exc:
        _deny("invalid external witness signature: " + type(exc).__name__)
    return {
        "external_highwater_sequence": sequence,
        "external_witness_receipt_sha256":
            hashlib.sha256(canonical(response)).hexdigest(),
    }
