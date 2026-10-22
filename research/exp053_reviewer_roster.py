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

# Ed25519 compressed public keys are NOT validated by a 32-byte length check.
# In some OpenSSL versions a low-order/identity public key verifies a
# signature without proving any independent private-key possession. These
# operations use only public coordinates; never process secret signing keys.
_FIELD = 2**255 - 19
_ORDER = 2**252 + 27742317777372353535851937790883648493
_D = (-121665 * pow(121666, _FIELD - 2, _FIELD)) % _FIELD
_SQRT_MINUS_ONE = pow(2, (_FIELD - 1) // 4, _FIELD)
_IDENTITY = (0, 1, 1, 0)


def _edwards_add(a, b):
    """Complete extended Edwards addition with a = -1, public inputs only."""
    x1, y1, z1, t1 = a
    x2, y2, z2, t2 = b
    left = ((y1 - x1) * (y2 - x2)) % _FIELD
    right = ((y1 + x1) * (y2 + x2)) % _FIELD
    cross = (2 * _D * t1 * t2) % _FIELD
    depth = (2 * z1 * z2) % _FIELD
    e = (right - left) % _FIELD
    f = (depth - cross) % _FIELD
    g = (depth + cross) % _FIELD
    h = (right + left) % _FIELD
    return ((e * f) % _FIELD, (g * h) % _FIELD,
            (f * g) % _FIELD, (e * h) % _FIELD)


def _prime_order_ed25519_public_key(raw):
    """True iff this canonical compressed point is nonzero and L-torsion-free.

    No external library's from_public_bytes() routine is assumed to enforce
    curve/subgroup constraints. This performs public-input group arithmetic.
    """
    if not isinstance(raw, bytes) or len(raw) != 32:
        return False
    packed = int.from_bytes(raw, "little")
    sign = packed >> 255
    y = packed & ((1 << 255) - 1)
    if y >= _FIELD:
        return False
    y2 = (y * y) % _FIELD
    denominator = (_D * y2 + 1) % _FIELD
    if denominator == 0:
        return False
    x2 = ((y2 - 1) * pow(denominator, _FIELD - 2, _FIELD)) % _FIELD
    x = pow(x2, (_FIELD + 3) // 8, _FIELD)
    if x * x % _FIELD != x2:
        x = (x * _SQRT_MINUS_ONE) % _FIELD
    if x * x % _FIELD != x2:
        return False
    if x & 1 != sign:
        x = (_FIELD - x) % _FIELD
    if not x and sign:
        return False  # Noncanonical encoding of a point with x=0.
    point = (x, y, 1, (x * y) % _FIELD)
    if point == _IDENTITY:
        return False
    acc = _IDENTITY
    for bit in bin(_ORDER)[2:]:
        acc = _edwards_add(acc, acc)
        if bit == "1":
            acc = _edwards_add(acc, point)
    # Projective identity: X == 0, Y == Z, modulo field prime.
    return acc[0] % _FIELD == 0 and (acc[1] - acc[2]) % _FIELD == 0


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
                base64.b64encode(pub).decode() != info["public_key_b64"] or
                not _prime_order_ed25519_public_key(pub)):
            deny("invalid or low-order Ed25519 public key")
        if reviewer_id in used_ids or pub in used_keys:
            deny("reviewer identities and signing keys must be distinct")
        used_ids.add(reviewer_id)
        used_keys.add(pub)
        keys[reviewer_id] = info["public_key_b64"]
    return {"slots": slots, "keys": keys, "roster_sha256": pinned_sha256,
            "owner_custody_reference_sha256": roster["owner_custody_reference_sha256"]}
