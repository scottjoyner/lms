#!/usr/bin/env python3
"""Fail-closed external witness-side operator authorization verification.

No signing and no production backend. Verifies pre-authorized source-free
operator receipts before any external CAS state mutation.
"""
import base64
import hashlib
from pathlib import Path

from exp053_execution_attestation import unambiguous_json
from exp053_race_review import canonical, deny, filehash

SPEC = Path(__file__).resolve().parent / "EXP053-AUTHENTICATED-WITNESS-REQUEST-SPEC-2026-10-17.md"
SPEC_SHA256 = "8bc1c1c03ae5672c86e8df5dc1112e53a233902c32f2811fcaceb237f7f299a8"
REQUEST_SCHEMA = "exp053-external-highwater-cas-request-v1"
AUTH_SCHEMA = "exp053-operator-run-authorization-v1"
REQUEST_FIELDS = {
    "schema", "nonce_hex", "sequence", "expected_previous_sequence",
    "previous_claim_sha256", "claim_sha256", "runner_execution_id_sha256",
    "operator_authorization_b64",
}
AUTH_FIELDS = {
    "schema", "runner_execution_id_sha256", "execution_attestation_sha256",
    "packet_sha256", "score_archive_sha256",
    "operator_custody_reference_sha256", "sequence", "previous_claim_sha256",
    "operator_public_key_b64", "signature_hex",
}
MAX_AUTH_BYTES = 16384
MAX_AUTH_B64_BYTES = 4 * ((MAX_AUTH_BYTES + 2) // 3)


def deny_witness(message):
    deny("EXP053 AUTHENTICATED WITNESS HOLD: " + message)


def hex64(value):
    return (isinstance(value, str) and len(value) == 64 and
            all(c in "0123456789abcdef" for c in value))


def verify_authenticated_request(request, independent_operator_key_sha256):
    """Verify bytes+Ed25519 binding before witness CAS BEGIN IMMEDIATE."""
    if not SPEC.is_file() or filehash(SPEC) != SPEC_SHA256:
        deny_witness("frozen operator-authenticated witness supplement changed")
    if not hex64(independent_operator_key_sha256):
        deny_witness("external witness has no independently trusted operator key")
    if not isinstance(request, dict) or set(request) != REQUEST_FIELDS:
        deny_witness("unapproved external CAS request shape")
    if request["schema"] != REQUEST_SCHEMA:
        deny_witness("wrong external CAS request schema")
    sequence = request["sequence"]
    if (type(sequence) is not int or sequence < 1 or
            type(request["expected_previous_sequence"]) is not int or
            request["expected_previous_sequence"] != sequence - 1):
        deny_witness("invalid CAS sequence or predecessor")
    for name in ("nonce_hex", "previous_claim_sha256", "claim_sha256",
                 "runner_execution_id_sha256"):
        if not hex64(request[name]):
            deny_witness("invalid CAS SHA256/nonce field " + name)
    encoded = request["operator_authorization_b64"]
    if (not isinstance(encoded, str) or
            not 1 <= len(encoded) <= MAX_AUTH_B64_BYTES):
        deny_witness("missing or unbounded signed operator payload")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError) as exc:
        deny_witness("bad operator payload encoding: " + type(exc).__name__)
    if (not 1 <= len(raw) <= MAX_AUTH_BYTES or
            base64.b64encode(raw).decode() != encoded or
            hashlib.sha256(raw).hexdigest() != request["claim_sha256"]):
        deny_witness("operator authorization exact bytes do not match CAS claim")
    signed = unambiguous_json(raw)
    if not isinstance(signed, dict) or set(signed) != AUTH_FIELDS:
        deny_witness("unexpected operator authorization fields")
    if signed["schema"] != AUTH_SCHEMA:
        deny_witness("operator authorization schema mismatch")
    for inner, outer in (("sequence", "sequence"),
                         ("previous_claim_sha256", "previous_claim_sha256"),
                         ("runner_execution_id_sha256", "runner_execution_id_sha256")):
        if type(signed[inner]) != type(request[outer]) or signed[inner] != request[outer]:
            deny_witness("operator signature bound to wrong " + inner)
    for name in ("execution_attestation_sha256", "packet_sha256",
                 "score_archive_sha256", "operator_custody_reference_sha256"):
        if not hex64(signed[name]):
            deny_witness("missing independently signed " + name)
    try:
        public = base64.b64decode(signed["operator_public_key_b64"], validate=True)
        if (len(public) != 32 or
                base64.b64encode(public).decode() != signed["operator_public_key_b64"] or
                hashlib.sha256(public).hexdigest() != independent_operator_key_sha256):
            deny_witness("unauthorized or noncanonical operator public key")
        signature = bytes.fromhex(signed["signature_hex"])
        if len(signature) != 64:
            deny_witness("malformed operator signature")
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        unsigned = {k: v for k, v in signed.items() if k != "signature_hex"}
        Ed25519PublicKey.from_public_bytes(public).verify(signature, canonical(unsigned))
    except Exception as exc:
        deny_witness("operator signature verification failed: " + type(exc).__name__)
    return {"operator_key_sha256": independent_operator_key_sha256,
            "claim_sha256": request["claim_sha256"],
            "execution_id_sha256": request["runner_execution_id_sha256"]}
