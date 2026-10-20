#!/usr/bin/env python3
"""Future-only scorer evidence verification; DOES NOT execute or import models.

A valid Ed25519 signature authenticates a key holder, not the truthfulness of
their numerical outputs. Both independent operator pins remain EMPTY.
"""
import base64
import hashlib
import json
import os
import re
import stat
from pathlib import Path

from exp053_race_review import canonical, filehash, deny

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "research" / "EXP053-EXECUTION-ATTESTATION-SPEC-2026-10-17.md"
SPEC_SHA256 = "3564f784ec36f9e060b2b55b5375c07589595fa73f289a38caafa2865ca285b6"
SCHEMA = "exp053-execution-attestation-v1"
# UNCONFIGURED until independent human operator approves exact runner custody.
APPROVED_RUNNER_PUBLIC_KEY_SHA256 = ""
APPROVED_RUNNER_SOURCE_SHA256 = ""
MAX_ATTESTATION_BYTES = 16384
MAX_SCORE_ARCHIVE_BYTES = 32 * 1024 * 1024
FIELDS = {
    "schema", "execution_id", "runner_public_key_b64",
    "runner_source_sha256", "packet_sha256", "signed_admission_sha256",
    "weight_inventory_sha256", "tokenizer_preflight_sha256",
    "prompt_builder_sha256", "score_archive_sha256", "attempt_log_sha256",
    "checkpoint_receipts", "score_row_count", "execution_environment_sha256",
    "runner_mode", "simulation", "signature_hex",
}


def hex_digest(s):
    return (isinstance(s, str) and len(s) == 64 and
            all(ch in "0123456789abcdef" for ch in s))


def private_regular_file(path, private_dir, *, small_limit=None):
    """Read private evidence from one safe fd, refusing symlink ancestors.

    Preserve incremental hashing for large score and attempt-log files;
    only small bounded JSON evidence is retained as in-memory bytes.
    """
    from exp053_reviewer_roster import _open_no_symlink_ancestors

    if not all(getattr(os, flag, 0) for flag in
               ("O_PATH", "O_DIRECTORY", "O_NOFOLLOW", "O_NONBLOCK")):
        deny("secure execution evidence pathname traversal unavailable")
    private_root = Path(os.path.abspath(os.fspath(private_dir)))
    path = Path(os.path.abspath(os.fspath(path)))
    if path == private_root or not path.is_relative_to(private_root):
        deny("execution evidence must reside inside selected private directory")
    if (small_limit is not None and
            (type(small_limit) is not int or small_limit < 1)):
        deny("execution evidence size limit invalid")

    try:
        fd = _open_no_symlink_ancestors(path)
        try:
            before = os.fstat(fd)
            if (not stat.S_ISREG(before.st_mode) or before.st_mode & 0o077 or
                    before.st_nlink != 1 or before.st_uid != os.geteuid() or
                    before.st_size < 1):
                deny("execution evidence is not a private owner regular file")
            if small_limit is not None and before.st_size > small_limit:
                deny("execution attestation exceeds bounded JSON size")
            digest = hashlib.sha256()
            data = bytearray() if small_limit is not None else None
            size = 0
            while True:
                block = os.read(fd, 1 << 20)
                if not block:
                    break
                size += len(block)
                if small_limit is not None and size > small_limit:
                    deny("execution attestation unexpectedly exceeded limit")
                digest.update(block)
                if data is not None:
                    data.extend(block)
            after = os.fstat(fd)
            def identity(st):
                return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns,
                        st.st_ctime_ns)
            if identity(before) != identity(after) or size != before.st_size:
                deny("execution evidence mutated while hashing")
            # Check the opened fd is still the inode reached by the caller's
            # ORIGINAL lexical path, not a symlink-resolved substitute.
            reopened = _open_no_symlink_ancestors(path)
            try:
                current = os.fstat(reopened)
                if (current.st_dev, current.st_ino) != (before.st_dev, before.st_ino):
                    deny("execution evidence pathname/inode changed during hashing")
            finally:
                os.close(reopened)
            return digest.hexdigest(), bytes(data) if data is not None else None
        finally:
            os.close(fd)
    except (OSError, RuntimeError) as exc:
        deny("execution evidence cannot be securely opened: " + type(exc).__name__)


def unambiguous_json(data):
    def reject_duplicates(pairs):
        output = {}
        for k, v in pairs:
            if k in output:
                raise ValueError("duplicate execution-attestation JSON property")
            output[k] = v
        return output
    def reject_nonfinite(value):
        raise ValueError("non-finite JSON value: " + value)
    try:
        return json.loads(data, object_pairs_hook=reject_duplicates,
                          parse_constant=reject_nonfinite)
    except (ValueError, UnicodeDecodeError) as exc:
        deny("invalid or ambiguous execution-attestation JSON: " + type(exc).__name__)


def read_attested_scores(score_archive, private_dir, score_sha256):
    """Bind parsed score bytes to the SAME descriptor that produced their hash.

    Never accept a pre- or post-read path hash as proof of what JSON was
    actually parsed. A swapped file cannot be silently substituted between
    path hashing and metrics analysis. This does not authenticate a runner.
    """
    if not hex_digest(score_sha256):
        deny("expected score archive fingerprint is not approved")
    observed, data = private_regular_file(
        score_archive, private_dir, small_limit=MAX_SCORE_ARCHIVE_BYTES)
    if observed != score_sha256:
        deny("parsed score-file bytes differ from signed execution receipt")
    rows = unambiguous_json(data)
    if not isinstance(rows, list):
        deny("private score archive must contain a JSON list")
    return rows


def verify_attestation(attestation_path, score_archive, attempt_log, private_dir,
                       *, packet_sha256, admission_sha256, weight_inventory_sha256,
                       tokenizer_preflight_sha256, prompt_builder_sha256,
                       checkpoint_receipts, approved_count):
    """Verify future private score+attempt byte custody and operator-pinned signer.

    This is NOT a signer enrollment helper and never produces a signature.
    """
    if not SPEC.is_file() or filehash(SPEC) != SPEC_SHA256:
        deny("frozen prospective execution-attestation protocol drift")
    if (not hex_digest(APPROVED_RUNNER_PUBLIC_KEY_SHA256) or
            not hex_digest(APPROVED_RUNNER_SOURCE_SHA256)):
        deny("no independent operator-approved scorer key and source pin")
    for label, value in (
        ("packet", packet_sha256), ("human admission", admission_sha256),
        ("weights", weight_inventory_sha256),
        ("tokenizer", tokenizer_preflight_sha256),
        ("prompt code", prompt_builder_sha256),
    ):
        if not hex_digest(value):
            deny(label + " evidence digest is invalid")
    if type(approved_count) is not int or approved_count < 40:
        deny("fewer than 40 authentic independently admitted articles")
    if (not isinstance(checkpoint_receipts, dict) or len(checkpoint_receipts) != 3 or
            not all(isinstance(k, str) and hex_digest(v)
                    for k, v in checkpoint_receipts.items())):
        deny("untrusted checkpoint identity inventory")
    if len({str(Path(p).resolve()) for p in
            (attestation_path, score_archive, attempt_log)}) != 3:
        deny("execution attestation, score archive, attempt log must be distinct")
    score_sha, _ = private_regular_file(score_archive, private_dir)
    attempts_sha, _ = private_regular_file(attempt_log, private_dir)
    attestation_sha, raw = private_regular_file(
        attestation_path, private_dir, small_limit=MAX_ATTESTATION_BYTES)
    signed = unambiguous_json(raw)
    if not isinstance(signed, dict) or set(signed) != FIELDS:
        deny("missing, extra, or private-leaking execution-attestation fields")
    if signed["schema"] != SCHEMA:
        deny("execution-attestation schema mismatch")
    if (not isinstance(signed["execution_id"], str) or
            not re.fullmatch(r"[A-Za-z0-9_-]{16,128}", signed["execution_id"])):
        deny("invalid opaque scorer run identifier")
    if signed["simulation"] is not False or signed["runner_mode"] != "raw":
        deny("simulation or non-raw inference receipt cannot be admitted")
    expected = {
        "packet_sha256": packet_sha256,
        "signed_admission_sha256": admission_sha256,
        "weight_inventory_sha256": weight_inventory_sha256,
        "tokenizer_preflight_sha256": tokenizer_preflight_sha256,
        "prompt_builder_sha256": prompt_builder_sha256,
        "runner_source_sha256": APPROVED_RUNNER_SOURCE_SHA256,
        "score_archive_sha256": score_sha,
        "attempt_log_sha256": attempts_sha,
        "checkpoint_receipts": checkpoint_receipts,
        "score_row_count": approved_count * 3 * 3 * 2,
    }
    for field, value in expected.items():
        if signed[field] != value or type(signed[field]) != type(value):
            deny("execution receipt not bound to approved " + field)
    if not hex_digest(signed["execution_environment_sha256"]):
        deny("missing private execution environment fingerprint")
    try:
        raw_key = base64.b64decode(signed["runner_public_key_b64"], validate=True)
        if (len(raw_key) != 32 or
                base64.b64encode(raw_key).decode() != signed["runner_public_key_b64"] or
                hashlib.sha256(raw_key).hexdigest() !=
                APPROVED_RUNNER_PUBLIC_KEY_SHA256):
            deny("runner public key does not match independently pinned operator root")
        signature = bytes.fromhex(signed["signature_hex"])
        if len(signature) != 64:
            deny("invalid Ed25519 execution-signature length")
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        payload = {k: v for k, v in signed.items() if k != "signature_hex"}
        Ed25519PublicKey.from_public_bytes(raw_key).verify(signature, canonical(payload))
    except Exception as exc:
        deny("execution-signature verification failed: " + type(exc).__name__)
    return {
        "schema": SCHEMA,
        "execution_id_sha256": hashlib.sha256(signed["execution_id"].encode()).hexdigest(),
        "score_archive_sha256": score_sha,
        "attempt_log_sha256": attempts_sha,
        "execution_attestation_sha256": attestation_sha,
        "runner_public_key_sha256": APPROVED_RUNNER_PUBLIC_KEY_SHA256,
        "runner_source_sha256": APPROVED_RUNNER_SOURCE_SHA256,
        "score_rows": signed["score_row_count"],
    }
