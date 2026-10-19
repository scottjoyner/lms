#!/usr/bin/env python3
"""Human-operated Ed25519 sign/export helper; never creates a review or private key."""
import argparse
import base64
import json
import math
import os
import stat
from pathlib import Path
from cryptography.exceptions import UnsupportedAlgorithm
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from exp053_race_review import canonical,decisions_for,deny,save_private
ATTEST=("I independently reviewed these source passages and responses; "
        "no model predictions or other reviewer labels were provided to me.")
MAX_KEY_BYTES = 16384


def get_key(path):
    """Load a reviewer-operated local PEM without following the key symlink.

    Presence/possession does NOT certify that a real independent human is
    enrolled. Do not print private bytes or their location on failure.
    """
    if not getattr(os, "O_NOFOLLOW", 0) or not getattr(os, "O_NONBLOCK", 0):
        deny("secure signing-key file access unavailable")
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
    try:
        fd = os.open(path, flags)
    except OSError:
        deny("signing key unavailable or symlinked")
    try:
        before = os.fstat(fd)
        if (not stat.S_ISREG(before.st_mode) or
                before.st_uid != os.geteuid() or
                before.st_mode & 0o077 or before.st_nlink != 1):
            deny("signing key must be a private owner-controlled regular file")
        if not 1 <= before.st_size <= MAX_KEY_BYTES:
            deny("signing key file size invalid")
        raw = os.read(fd, MAX_KEY_BYTES + 1)
        after = os.fstat(fd)
        identity = lambda s: (s.st_dev, s.st_ino, s.st_size,
                              s.st_mtime_ns, s.st_ctime_ns)
        if len(raw) != before.st_size or identity(before) != identity(after):
            deny("signing key file changed while reading")
    except OSError:
        deny("signing key could not be inspected securely")
    finally:
        os.close(fd)
    try:
        key = serialization.load_pem_private_key(raw, password=None)
    except (ValueError, TypeError, UnsupportedAlgorithm) as exc:
        deny("invalid unencrypted Ed25519 signing key: " + type(exc).__name__)
    if not isinstance(key, Ed25519PrivateKey):
        deny("Ed25519 PEM private key required")
    return key
def public_b64(key):
    return base64.b64encode(key.public_key().public_bytes(
       encoding=serialization.Encoding.Raw,
       format=serialization.PublicFormat.Raw)).decode()
MAX_INPUT_BYTES = 8_000_000


def _unique_fields(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            deny("duplicate field in offline signer JSON")
        out[key] = value
    return out


def _invalid_constant(_value):
    deny("invalid numeric constant in offline signer JSON")


def _finite_float(value):
    parsed = float(value)
    if not math.isfinite(parsed):
        deny("non-finite numeric value in offline signer JSON")
    return parsed


def read_offline_json(path):
    """Read only one bounded, stable, non-symlinked local JSON file."""
    if not getattr(os, "O_NOFOLLOW", 0) or not getattr(os, "O_NONBLOCK", 0):
        deny("secure offline review input access unavailable")
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError:
        deny("offline signer input unavailable or symlinked")
    try:
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or not 1 <= info.st_size <= MAX_INPUT_BYTES:
                deny("offline signer input is not a bounded regular file")
            data = bytearray()
            while len(data) < MAX_INPUT_BYTES + 1:
                part = os.read(fd, min(65536, MAX_INPUT_BYTES + 1 - len(data)))
                if not part:
                    break
                data.extend(part)
            after = os.fstat(fd)
            identity = lambda st: (st.st_dev, st.st_ino, st.st_size,
                                   st.st_mtime_ns, st.st_ctime_ns)
            if len(data) != info.st_size or identity(info) != identity(after):
                deny("offline signer input changed while reading")
        except OSError:
            deny("offline signer input could not be inspected securely")
    finally:
        os.close(fd)
    try:
        result = json.loads(data.decode("utf-8"), object_pairs_hook=_unique_fields,
                            parse_constant=_invalid_constant,
                            parse_float=_finite_float)
    except (ValueError, UnicodeError, RecursionError):
        deny("offline signer input contains invalid or ambiguous JSON")
    if not isinstance(result, dict):
        deny("offline signer input must be a JSON object")
    return result


def sign(input_path, packet_path, key_path, out):
    input_path = Path(input_path)
    packet_path = Path(packet_path)
    out = Path(out)
    if input_path.resolve() == out.resolve() or out.exists():
        deny("signed output must be a new file")
    review = read_offline_json(input_path)
    packet = read_offline_json(packet_path)
    expected = {"schema", "reviewer_id", "role", "packet_sha256",
                "attestation", "decisions", "signature"}
    if set(review) != expected or review.get("schema") != "exp053-review-v1":
        deny("unknown or missing offline reviewer draft fields")
    rid = review.get("reviewer_id")
    if not isinstance(rid, str) or not rid:
        deny("reviewer_id must be set by the human reviewer")
    if review.get("signature") != "":
        deny("an already-signed review cannot be signed as a draft")
    if review.get("attestation") != ATTEST:
        deny("reviewer must personally attest prior to signing")
    if review.get("role") not in ("independent", "adjudicator"):
        deny("invalid reviewer role")
    decisions = review.get("decisions")
    if not isinstance(decisions, list):
        deny("malformed offline review decisions")
    scope = None
    if review["role"] == "adjudicator":
        scope = set()
        for d in decisions:
            if not isinstance(d, dict):
                deny("malformed offline adjudicator decision")
            cid = d.get("case_id")
            if not isinstance(cid, str) or not cid or cid in scope:
                deny("invalid or repeated offline adjudicator case ID")
            scope.add(cid)
        if not scope:
            deny("empty adjudicator review")
    decisions_for(review, packet, required_ids=scope)
    key = get_key(key_path)
    unsigned = {k: v for k, v in review.items() if k != "signature"}
    unsigned["signature"] = key.sign(canonical(unsigned)).hex()
    save_private(out, unsigned)
    print("SIGNED_REVIEW", unsigned["reviewer_id"], "PUBLIC_KEY_B64", public_b64(key))

def main():
    p=argparse.ArgumentParser()
    sub=p.add_subparsers(dest="action",required=True)
    s=sub.add_parser("sign")
    s.add_argument("--review",type=Path,required=True)
    s.add_argument("--packet",type=Path,required=True)
    s.add_argument("--private-key",type=Path,required=True)
    s.add_argument("--out",type=Path,required=True)
    k=sub.add_parser("public-key")
    k.add_argument("--private-key",type=Path,required=True)
    x=p.parse_args()
    if x.action=="public-key":print(public_b64(get_key(x.private_key)))
    else:sign(x.review,x.packet,x.private_key,x.out)
if __name__=="__main__":main()
