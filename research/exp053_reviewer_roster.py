#!/usr/bin/env python3
"""Fail-closed private human-reviewer enrollment validation.

This does NOT verify human identity. An independent study owner must establish
human independence and key custody out of band, then code-review and pin the
exact private roster SHA256. No approved roster currently exists.
"""
import base64
import hashlib
import json
import stat
from pathlib import Path

SCHEMA = "exp053-owner-reviewed-roster-v1"
ROLES = {"A": "independent", "B": "independent", "ADJ": "adjudicator"}


def deny(message):
    raise ValueError("HOLD EXP053 ROSTER: " + message)


def lower_sha(value):
    return (isinstance(value, str) and len(value) == 64 and
            all(c in "0123456789abcdef" for c in value))


def load_roster(path, packet_sha256, pinned_sha256, repo_root):
    """Validate exact owner-pinned private bytes and A/B/adjudicator slots."""
    if not lower_sha(pinned_sha256):
        deny("the owner has not independently approved any reviewer roster")
    path = Path(path)
    if path.is_symlink():
        deny("symlinked reviewer roster is not allowed")
    try:
        resolved = path.resolve(strict=True)
        if resolved.is_relative_to(Path(repo_root).resolve()):
            deny("private reviewer roster must not reside in the public repository")
        mode = path.stat().st_mode
        if not stat.S_ISREG(mode) or mode & 0o077:
            deny("roster must be a private regular file (mode 0600)")
        data = path.read_bytes()
    except (OSError, RuntimeError) as exc:
        deny("roster cannot be read: " + type(exc).__name__)
    if hashlib.sha256(data).hexdigest() != pinned_sha256:
        deny("private roster differs from owner-reviewed fingerprint")
    try:
        roster = json.loads(data)
    except (ValueError, UnicodeDecodeError) as exc:
        deny("invalid roster encoding: " + type(exc).__name__)
    required = {"schema", "packet_sha256", "owner_custody_reference_sha256",
                "blind_assignments_confirmed", "slots"}
    if not isinstance(roster, dict) or set(roster) != required:
        deny("unexpected roster fields")
    if roster["schema"] != SCHEMA or roster["packet_sha256"] != packet_sha256:
        deny("roster schema or sealed candidate packet mismatch")
    if not lower_sha(roster["owner_custody_reference_sha256"]):
        deny("missing independently maintained owner enrollment record fingerprint")
    if roster["blind_assignments_confirmed"] is not True:
        deny("blind-review assignments not owner confirmed")
    slots = roster["slots"]
    if not isinstance(slots, dict) or set(slots) not in ({"A", "B"}, {"A", "B", "ADJ"}):
        deny("expected distinct A/B slots and optional adjudicator")
    used_ids = set()
    used_keys = set()
    keys = {}
    for slot, info in slots.items():
        if not isinstance(info, dict) or set(info) != {
                "reviewer_id", "role", "public_key_b64"}:
            deny("invalid reviewer enrollment fields")
        reviewer_id = info["reviewer_id"]
        if (not isinstance(reviewer_id, str) or not 4 <= len(reviewer_id) <= 80 or
                not all(x.isalnum() or x in "_-" for x in reviewer_id)):
            deny("invalid pseudonymous reviewer identity")
        if info["role"] != ROLES[slot]:
            deny("reviewer role not approved for blinded slot")
        try:
            pub = base64.b64decode(info["public_key_b64"], validate=True)
        except (TypeError, ValueError) as exc:
            deny("bad reviewer key encoding: " + type(exc).__name__)
        if (len(pub) != 32 or
                base64.b64encode(pub).decode() != info["public_key_b64"]):
            deny("invalid Ed25519 public key")
        if reviewer_id in used_ids or pub in used_keys:
            deny("reviewer identities and signing keys must be distinct")
        used_ids.add(reviewer_id)
        used_keys.add(pub)
        keys[reviewer_id] = info["public_key_b64"]
    return {"slots": slots, "keys": keys, "roster_sha256": pinned_sha256,
            "owner_custody_reference_sha256": roster["owner_custody_reference_sha256"]}
