#!/usr/bin/env python3
"""Research-only operator-signed single-use EXP053 run-claim verifier.

No actual signing keys, scorer, model execution or operator ledger is
configured. A local claim alone CANNOT prove honest native model inference.
"""
import base64
import fcntl
import hashlib
import json
import os
import stat
from pathlib import Path

from exp053_execution_attestation import (
    hex_digest, private_regular_file, unambiguous_json,
)
from exp053_race_review import canonical, deny, filehash
from exp053_external_witness import commit_independent_highwater

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "research" / "EXP053-OPERATOR-RUN-UNIQUENESS-SPEC-2026-10-17.md"
SPEC_SHA256 = "8b3999c0febd6269d8a96f0ee31c719736c32aee95b769abc201f7beb51bada9"
MARKER_SPEC = ROOT / "research" / "EXP053-OPERATOR-RUN-MARKER-SUPPLEMENT-2026-10-17.md"
MARKER_SPEC_SHA256 = "fdccbe3b99fb11eba77a6f7f83c759a1b740e1a12c7e0cd07d4aeca29401049a"
SCHEMA = "exp053-operator-run-authorization-v1"
STATE_SCHEMA = "exp053-operator-claim-state-v1"
AUTH_FIELDS = {
    "schema", "runner_execution_id_sha256", "execution_attestation_sha256",
    "packet_sha256", "score_archive_sha256",
    "operator_custody_reference_sha256", "sequence", "previous_claim_sha256",
    "operator_public_key_b64", "signature_hex",
}
STATE_FIELDS = {"schema", "last_sequence", "last_claim_sha256"}
MAX_AUTH_BYTES = 16384
MAX_STATE_BYTES = 4096

# Intentionally unconfigured: these are NEVER supplied by CLI/user arguments.
# Each requires an independently approved, code-reviewed source change.
APPROVED_OPERATOR_PUBLIC_KEY_SHA256 = ""
APPROVED_OPERATOR_CLAIM_LEDGER_PATH = ""


def _deny_invalid(label):
    deny("EXP053 OPERATOR RUN HOLD: " + label)


def _signed_authorization(path, private_dir, provenance, packet_sha256,
                          scorer_key_sha256):
    """Read, verify and hash a private operator-signed run authorization."""
    digest, data = private_regular_file(path, private_dir, small_limit=MAX_AUTH_BYTES)
    record = unambiguous_json(data)
    if not isinstance(record, dict) or set(record) != AUTH_FIELDS:
        _deny_invalid("unknown or missing operator authorization fields")
    if record["schema"] != SCHEMA:
        _deny_invalid("unknown operator authorization schema")
    if (type(record["sequence"]) is not int or record["sequence"] < 1 or
            not hex_digest(record["previous_claim_sha256"]) or
            not hex_digest(record["operator_custody_reference_sha256"])):
        _deny_invalid("operator sequence or independent custody reference malformed")
    expected = {
        "runner_execution_id_sha256": provenance["execution_id_sha256"],
        "execution_attestation_sha256": provenance["execution_attestation_sha256"],
        "packet_sha256": packet_sha256,
        "score_archive_sha256": provenance["score_archive_sha256"],
    }
    for field, value in expected.items():
        if not hex_digest(value) or record[field] != value or type(record[field]) is not str:
            _deny_invalid("operator authorization mismatches verified " + field)
    try:
        key = base64.b64decode(record["operator_public_key_b64"], validate=True)
        if (len(key) != 32 or base64.b64encode(key).decode() !=
                record["operator_public_key_b64"] or
                hashlib.sha256(key).hexdigest() !=
                APPROVED_OPERATOR_PUBLIC_KEY_SHA256 or
                hashlib.sha256(key).hexdigest() == scorer_key_sha256):
            _deny_invalid("operator key is unapproved or matches scorer")
        signature = bytes.fromhex(record["signature_hex"])
        if len(signature) != 64:
            _deny_invalid("operator signature has invalid length")
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        unsigned = {k: v for k, v in record.items() if k != "signature_hex"}
        Ed25519PublicKey.from_public_bytes(key).verify(signature, canonical(unsigned))
    except Exception as exc:
        _deny_invalid("operator signature/key invalid: " + type(exc).__name__)
    return record, digest, data


def _pinned_operator_parent(parent_path):
    """Open every lexical directory component without following symlinks."""
    from exp053_reviewer_roster import _open_no_symlink_ancestors
    if not all(getattr(os, flag, 0) for flag in
               ("O_PATH", "O_DIRECTORY", "O_NOFOLLOW", "O_NONBLOCK")):
        _deny_invalid("secure operator journal pathname traversal unavailable")
    fd = _open_no_symlink_ancestors(parent_path)
    try:
        info = os.fstat(fd)
        if (not stat.S_ISDIR(info.st_mode) or
                info.st_uid != os.geteuid() or info.st_mode & 0o022):
            _deny_invalid("operator journal parent owner, mode, or type invalid")
    except BaseException:
        os.close(fd)
        raise
    return fd


def _require_operator_parent_stable(parent_path, pinned_fd):
    """Refuse a parent path redirect before or after remote claim CAS."""
    fresh = _pinned_operator_parent(parent_path)
    try:
        actual = os.fstat(fresh)
        approved = os.fstat(pinned_fd)
        if (actual.st_dev, actual.st_ino) != (approved.st_dev, approved.st_ino):
            _deny_invalid("operator journal parent pathname changed")
    finally:
        os.close(fresh)


def _consume_claim(path, private_dir, authorization, authorization_sha256,
                   *, operator_public_key_sha256, scorer_public_key_sha256,
                   operator_authorization_bytes):
    """Atomically serialize claims on one pre-existing human-owned state inode.

    Crash damage always fails closed. The operator must keep an independent
    backup/high-water witness: a filesystem snapshot rollback is out of scope.
    """
    path = Path(path)
    if not path.is_absolute() or path.is_symlink():
        _deny_invalid("operator claim journal must be an approved absolute file")
    try:
        resolved = path.resolve(strict=True)
        root_dir = ROOT.resolve()
        scorer_dir = Path(private_dir).resolve(strict=True)
        if (resolved.is_relative_to(root_dir) or
                resolved.is_relative_to(scorer_dir)):
            _deny_invalid("operator journal is not separated from scorer and source")
        parent_fd = _pinned_operator_parent(path.parent)
        try:
            parent = os.fstat(parent_fd)
            if (not stat.S_ISDIR(parent.st_mode) or
                    parent.st_uid != os.geteuid() or parent.st_mode & 0o022):
                _deny_invalid("operator journal parent not securely owner-controlled")
            fd = os.open(path.name, os.O_RDWR | os.O_NONBLOCK | os.O_NOFOLLOW,
                         dir_fd=parent_fd)
            try:
                with os.fdopen(fd, "r+b", closefd=False) as stream:
                    fcntl.flock(fd, fcntl.LOCK_EX)
                    before = os.fstat(fd)
                    linked = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
                    if (not stat.S_ISREG(before.st_mode) or
                            before.st_mode & 0o077 or before.st_nlink != 1 or
                            before.st_uid != os.getuid() or
                            (before.st_dev, before.st_ino) !=
                            (linked.st_dev, linked.st_ino) or
                            not 1 <= before.st_size <= MAX_STATE_BYTES):
                        _deny_invalid("operator journal ownership, mode, or inode invalid")
                    raw = stream.read(MAX_STATE_BYTES + 1)
                    after = os.fstat(fd)
                    if (len(raw) != before.st_size or len(raw) > MAX_STATE_BYTES or
                            (before.st_ino, before.st_dev, before.st_size,
                             before.st_mtime_ns, before.st_ctime_ns) !=
                            (after.st_ino, after.st_dev, after.st_size,
                             after.st_mtime_ns, after.st_ctime_ns)):
                        _deny_invalid("operator journal changed during locked read")
                    state = unambiguous_json(raw)
                    if (not isinstance(state, dict) or set(state) != STATE_FIELDS or
                            state["schema"] != STATE_SCHEMA or
                            type(state["last_sequence"]) is not int or
                            state["last_sequence"] < 0 or
                            not hex_digest(state["last_claim_sha256"]) or
                            (state["last_sequence"] == 0 and
                             state["last_claim_sha256"] != "0" * 64)):
                        _deny_invalid("operator state missing, corrupted, or manually reset")
                    if (authorization["sequence"] != state["last_sequence"] + 1 or
                            authorization["previous_claim_sha256"] !=
                            state["last_claim_sha256"]):
                        _deny_invalid("authorization already used, stale, or wrong chained sequence")
                    # A separately administered CAS must commit first: a restored
                    # local journal cannot present a fresh high-water response.
                    # If remote commit succeeds and local update fails, HOLD.
                    _require_operator_parent_stable(path.parent, parent_fd)
                    witness_evidence = commit_independent_highwater(
                        sequence=authorization["sequence"],
                        previous_claim_sha256=authorization["previous_claim_sha256"],
                        claim_sha256=authorization_sha256,
                        execution_id_sha256=authorization["runner_execution_id_sha256"],
                        operator_public_key_sha256=operator_public_key_sha256,
                        scorer_public_key_sha256=scorer_public_key_sha256,
                        operator_authorization_bytes=operator_authorization_bytes)
                    new_state = {
                        "schema": STATE_SCHEMA,
                        "last_sequence": authorization["sequence"],
                        "last_claim_sha256": authorization_sha256,
                    }
                    data = json.dumps(new_state, sort_keys=True,
                                      separators=(",", ":")).encode() + b"\n"
                    # Independent CAS may have committed; changed path => HOLD.
                    _require_operator_parent_stable(path.parent, parent_fd)
                    # Recheck same path/inode before creating a durable claim.
                    linked = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
                    if (linked.st_ino, linked.st_dev) != (before.st_ino, before.st_dev):
                        _deny_invalid("operator claim journal replaced while locked")
                    marker_id = authorization["runner_execution_id_sha256"]
                    marker_path = path.with_name(path.name + ".used-" + marker_id)
                    try:
                        marker_fd = os.open(
                            marker_path.name, os.O_WRONLY | os.O_CREAT |
                            os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=parent_fd)
                    except FileExistsError:
                        _deny_invalid("scorer execution ID was already claimed, even with a new sequence")
                    try:
                        os.write(marker_fd, (authorization_sha256 + "\n").encode())
                        os.fsync(marker_fd)
                    finally:
                        os.close(marker_fd)
                    os.fsync(parent_fd)
                    # If interrupted after marker creation, refuse later reuse.
                    stream.seek(0)
                    stream.write(data)
                    stream.truncate()
                    stream.flush()
                    os.fsync(fd)
                    _require_operator_parent_stable(path.parent, parent_fd)
                    return {
                        "operator_sequence": authorization["sequence"],
                        "operator_authorization_sha256": authorization_sha256,
                        **witness_evidence,
                    }
            finally:
                os.close(fd)
        finally:
            os.close(parent_fd)
    except (OSError, RuntimeError) as exc:
        _deny_invalid("operator journal unavailable/cannot be locked: " +
                      type(exc).__name__)


def consume_unique_run(authorization_path, private_dir, *,
                       provenance, packet_sha256):
    """Single-use independent operator claim; call ONLY after all other checks.

    Empty operator keys/path intentionally prevent any real experiment run.
    """
    if (not SPEC.is_file() or filehash(SPEC) != SPEC_SHA256 or
            not MARKER_SPEC.is_file() or
            filehash(MARKER_SPEC) != MARKER_SPEC_SHA256):
        _deny_invalid("frozen prospective operator/marker protocols changed")
    if (not hex_digest(APPROVED_OPERATOR_PUBLIC_KEY_SHA256) or
            not isinstance(APPROVED_OPERATOR_CLAIM_LEDGER_PATH, str) or
            not APPROVED_OPERATOR_CLAIM_LEDGER_PATH):
        _deny_invalid("no owner-approved operator key or single-use journal")
    if (not isinstance(provenance, dict) or
            not hex_digest(provenance.get("runner_public_key_sha256")) or
            APPROVED_OPERATOR_PUBLIC_KEY_SHA256 ==
            provenance["runner_public_key_sha256"] or
            not hex_digest(packet_sha256)):
        _deny_invalid("operator root cannot substitute for scorer/candidate identity")
    authorization, digest, signed_bytes = _signed_authorization(
        authorization_path, private_dir, provenance, packet_sha256,
        provenance["runner_public_key_sha256"])
    return _consume_claim(
        APPROVED_OPERATOR_CLAIM_LEDGER_PATH, private_dir,
        authorization, digest,
        operator_public_key_sha256=APPROVED_OPERATOR_PUBLIC_KEY_SHA256,
        scorer_public_key_sha256=provenance["runner_public_key_sha256"],
        operator_authorization_bytes=signed_bytes)
