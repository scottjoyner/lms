#!/usr/bin/env python3
"""Fail-closed private human-reviewer enrollment validation.

This does NOT verify human identity. An independent study owner must establish
human independence and key custody out of band, then code-review and pin the
exact private roster SHA256. No approved roster currently exists.
"""
import base64
import hashlib
import json
import os
import stat
from pathlib import Path

SCHEMA = "exp053-owner-reviewed-roster-v1"
MAX_ROSTER_BYTES = 16384  # Private human-key metadata, never corpus source text.
ROLES = {"A": "independent", "B": "independent", "ADJ": "adjudicator"}


def deny(message):
    raise ValueError("HOLD EXP053 ROSTER: " + message)


def lower_sha(value):
    return (isinstance(value, str) and len(value) == 64 and
            all(c in "0123456789abcdef" for c in value))


def _open_no_symlink_ancestors(path):
    """Resolve each lexical path component via non-following dir-fds.

    Opening the final leaf with O_NOFOLLOW does NOT protect ancestor symlink
    substitutions. Every directory component is therefore independently
    checked by the kernel, including components under the public repo.
    """
    absolute = Path(os.path.abspath(os.fspath(path)))
    parts = absolute.parts
    if len(parts) < 2 or parts[0] != "/":
        deny("invalid private reviewer roster pathname")
    dflags = os.O_PATH | os.O_DIRECTORY | os.O_NOFOLLOW
    parent = os.open("/", dflags)
    leaf = None
    try:
        for component in parts[1:-1]:
            next_dir = os.open(component, dflags, dir_fd=parent)
            try:
                os.close(parent)
            except BaseException:
                # Ownership of next_dir has not been transferred. If the old
                # parent cannot be released, never strand the newly opened fd.
                os.close(next_dir)
                raise
            parent = next_dir
        leaf = os.open(parts[-1], os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW,
                       dir_fd=parent)
        return leaf
    finally:
        # Close(2) failures are ambiguous: on Linux EINTR may have already
        # released the descriptor. A blind retry could close a reused fd.
        try:
            parent_identity = os.fstat(parent)
        except OSError:
            parent_identity = None
        try:
            os.close(parent)
        except BaseException:
            # The caller did not receive the leaf on this error path.
            if leaf is not None:
                try:
                    os.close(leaf)
                except OSError:
                    pass
            if parent_identity is not None:
                try:
                    current = os.fstat(parent)
                    if (current.st_dev, current.st_ino) == (
                            parent_identity.st_dev, parent_identity.st_ino):
                        os.close(parent)  # Only retry if same opened inode.
                except OSError:
                    pass
            raise


def load_roster(path, packet_sha256, pinned_sha256, repo_root):
    """Validate exact owner-pinned private bytes and A/B/adjudicator slots."""
    if not lower_sha(pinned_sha256):
        deny("the owner has not independently approved any reviewer roster")
    if not all(getattr(os, name, 0) for name in
               ("O_PATH", "O_DIRECTORY", "O_NOFOLLOW", "O_NONBLOCK")):
        deny("secure nonfollowing roster file access unavailable")
    path = Path(path)
    absolute = Path(os.path.abspath(os.fspath(path)))
    try:
        public_root = Path(repo_root).resolve(strict=True)
        intended = absolute.resolve(strict=True)
        if intended.is_relative_to(public_root):
            deny("private reviewer roster must not reside in the public repository")
        fd = _open_no_symlink_ancestors(absolute)
        try:
            before = os.fstat(fd)
            if (not stat.S_ISREG(before.st_mode) or before.st_uid != os.geteuid() or
                    before.st_nlink != 1 or before.st_mode & 0o077):
                deny("roster must be an owner-private, single-link regular file")
            if not 1 <= before.st_size <= MAX_ROSTER_BYTES:
                deny("private reviewer roster size outside allowed limit")
            data = bytearray()
            while len(data) < MAX_ROSTER_BYTES + 1:
                block = os.read(fd, min(65536, MAX_ROSTER_BYTES + 1 - len(data)))
                if not block:
                    break
                data.extend(block)
            after = os.fstat(fd)
            def identity(st):
                return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns,
                        st.st_ctime_ns)
            if identity(before) != identity(after) or len(data) != before.st_size:
                deny("private roster changed during admission")
            # A parent swapped to a symlink, or a different same-byte roster,
            # must not be silently treated as the approved local file.
            if absolute.resolve(strict=True) != intended or intended.is_relative_to(public_root):
                deny("private roster pathname changed during admission")
            final_fd = _open_no_symlink_ancestors(absolute)
            try:
                final = os.fstat(final_fd)
                if (final.st_dev, final.st_ino) != (before.st_dev, before.st_ino):
                    deny("private roster inode changed during admission")
            finally:
                os.close(final_fd)
        finally:
            os.close(fd)
    except (OSError, RuntimeError) as exc:
        deny("roster cannot be read safely: " + type(exc).__name__)
    if hashlib.sha256(data).hexdigest() != pinned_sha256:
        deny("private roster differs from owner-reviewed fingerprint")
    def reject_duplicate_fields(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj:
                raise ValueError("duplicate key in owner-reviewed reviewer roster")
            obj[key] = value
        return obj
    try:
        roster = json.loads(data, object_pairs_hook=reject_duplicate_fields)
    except (ValueError, UnicodeDecodeError) as exc:
        deny("invalid or ambiguous roster encoding: " + type(exc).__name__)
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
