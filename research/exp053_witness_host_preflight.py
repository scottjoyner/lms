#!/usr/bin/env python3
"""Explicit-alias read-only off-host witness preflight. NO deployment/inference.

This script does not provision independent witness authority. Its ONLY remote
action is a short, allowlisted read-only Python diagnostic using pinned SSH
host keys and existing noninteractive authorization. It never opens a server.
"""
import argparse
import hashlib
import json
import re
import secrets
import shlex
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

SPEC = Path(__file__).resolve().parent / "EXP053-OFFHOST-READONLY-PREFLIGHT-SPEC-2026-10-17.md"
SPEC_SHA256 = "523cbece51e6d966e868e859e728946fe7c15990e83f428143d1a08fb5dceeb5"
SCHEMA = "exp053-offhost-readonly-preflight-v1"
REMOTE_FIELDS = {
    "challenge", "python", "sqlite", "cryptography_available",
    "openssl_available", "tpm_device_present", "filesystem",
    "free_bytes", "clock_unix",
}
HOST_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,62}\Z")
MAX_OUTPUT = 4096
BASE_DECISIONS = {
    "independent_administration_approved": False,
    "operator_key_custody_approved": False,
    "witness_durability_approved": False,
    "scientific_execution_authorized": False,
    "decision": "NO_GO_RESEARCH_ONLY",
}
REMOTE = '''
import importlib.util
import json
import os
import platform
import shutil
import sqlite3
import subprocess
import sys
import time

fs = subprocess.run(
    ["findmnt", "-n", "-o", "FSTYPE", "--target", "/"],
    capture_output=True, text=True, timeout=2, check=True)
root = os.statvfs("/")
print(json.dumps({
    "challenge": CHALLENGE_VALUE,
    "python": platform.python_version(),
    "sqlite": sqlite3.sqlite_version,
    "cryptography_available": importlib.util.find_spec("cryptography") is not None,
    "openssl_available": shutil.which("openssl") is not None,
    "tpm_device_present": (os.path.exists("/dev/tpmrm0") or os.path.exists("/dev/tpm0")),
    "filesystem": fs.stdout.strip(),
    "free_bytes": int(root.f_bavail * root.f_frsize),
    "clock_unix": time.time(),
}, separators=(",", ":")))
'''


def unambiguous_json(raw):
    def pairs_no_duplicates(pairs):
        out = {}
        for k, value in pairs:
            if k in out:
                raise ValueError("duplicate remote report field")
            out[k] = value
        return out

    def reject_constant(value):
        raise ValueError("nonfinite remote field: " + value)

    return json.loads(raw, object_pairs_hook=pairs_no_duplicates,
                      parse_constant=reject_constant)


def parse_remote(stdout, challenge):
    if not isinstance(stdout, str) or len(stdout) > MAX_OUTPUT:
        raise ValueError("oversized or missing remote witness preflight")
    data = unambiguous_json(stdout)
    if not isinstance(data, dict) or set(data) != REMOTE_FIELDS:
        raise ValueError("missing, extra, or private remote report fields")
    if data["challenge"] != challenge:
        raise ValueError("stale or replayed remote challenge")
    for k in ("python", "sqlite"):
        if (not isinstance(data[k], str) or
                not re.fullmatch(r"\d+\.\d+(?:\.\d+)?", data[k])):
            raise ValueError("untrusted Python/SQLite version field")
    for key in ("cryptography_available", "openssl_available",
                "tpm_device_present"):
        if type(data[key]) is not bool:
            raise ValueError("untrusted boolean capability field")
    if data["filesystem"] not in {"ext4", "btrfs", "xfs", "zfs"}:
        raise ValueError("unexpected or unsupported filesystem")
    if (type(data["free_bytes"]) is not int or
            data["free_bytes"] < 0 or
            data["free_bytes"] > (1 << 64)):
        raise ValueError("free space field malformed")
    if (type(data["clock_unix"]) not in (int, float) or
            not 1_600_000_000 <= data["clock_unix"] < 2_500_000_000):
        raise ValueError("remote clock field malformed")
    return data


def probe(alias, *, runner=subprocess.run, challenge_source=secrets.token_hex):
    if not isinstance(alias, str) or not HOST_RE.fullmatch(alias) or alias.startswith("-"):
        raise ValueError("invalid explicit SSH host alias")
    # Fixed remote code only: no user-supplied remote command or arguments.
    challenge = challenge_source(16)
    if not isinstance(challenge, str) or not re.fullmatch(r"[0-9a-f]{32}", challenge):
        raise ValueError("invalid read-only random challenge")
    source = "CHALLENGE_VALUE = " + repr(challenge) + "\n" + REMOTE
    command = [
        "ssh", "-T", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
        "-o", "ForwardAgent=no", "-o", "ClearAllForwardings=yes",
        # Force a fresh SSH handshake. Existing multiplexed masters may have
        # negotiated host-key policy different from this pinned-key probe.
        "-o", "ControlMaster=no", "-o", "ControlPath=none",
        "-o", "UpdateHostKeys=no", "-o", "PermitLocalCommand=no",
        "-o", "ConnectTimeout=4", "-o", "ServerAliveInterval=3",
        alias, "python3 -c " + shlex.quote(source),
    ]
    before_wall = time.time()
    before = time.monotonic()
    result = dict(BASE_DECISIONS, alias=alias,
                  status="FAIL_CLOSED", clock_error_bound_ms=None,
                  round_trip_ms=None)
    try:
        proc = runner(command, capture_output=True, text=True, timeout=9,
                      check=False)
        after = time.monotonic()
        after_wall = time.time()
        if proc.returncode != 0:
            raise ValueError("SSH command failed or host-key check refused")
        remote = parse_remote(proc.stdout, challenge)
        rtt = max(0.0, after - before)
        midpoint = (before_wall + after_wall) / 2
        clock_error_bound = abs(remote["clock_unix"] - midpoint) + rtt / 2
        python_version = tuple(map(int, remote["python"].split(".")))
        sqlite_version = tuple(map(int, remote["sqlite"].split(".")))
        ready = (
            python_version >= (3, 10) and
            sqlite_version >= (3, 35) and
            remote["cryptography_available"] and
            remote["openssl_available"] and
            remote["free_bytes"] >= 2 * 1024 ** 3
        )
        status = ("READ_ONLY_CANDIDATE" if ready else "CAPABILITY_HOLD")
        if clock_error_bound > 10:
            status = "TIME_SKEW_HOLD"
        result.update(
            status=status,
            round_trip_ms=round(rtt * 1000, 1),
            clock_error_bound_ms=round(clock_error_bound * 1000, 1),
            python=remote["python"], sqlite=remote["sqlite"],
            cryptography_available=remote["cryptography_available"],
            openssl_available=remote["openssl_available"],
            tpm_device_present=remote["tpm_device_present"],
            filesystem=remote["filesystem"],
            free_gib=round(remote["free_bytes"] / (1024 ** 3), 1),
        )
    except (OSError, ValueError, subprocess.TimeoutExpired):
        # Do not leak stdout/stderr, file paths, IPs or machine IDs.
        result["status"] = "FAIL_CLOSED"
    return result


def main(argv=None, *, runner=subprocess.run):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("aliases", nargs="+", help="Existing pinned SSH host aliases")
    args = parser.parse_args(argv)
    if (not SPEC.is_file() or
            hashlib.sha256(SPEC.read_bytes()).hexdigest() != SPEC_SHA256):
        raise SystemExit("HOLD: prospective read-only witness protocol drift")
    aliases = args.aliases
    if len(aliases) != len(set(aliases)) or len(aliases) > 12:
        raise SystemExit("HOLD: duplicate or excessive explicit SSH aliases")
    results = [probe(alias, runner=runner) for alias in aliases]
    report = {
        "schema": SCHEMA,
        "observed_utc": datetime.now(timezone.utc).isoformat(),
        **BASE_DECISIONS,
        "hosts": results,
        "qualification_scope": "read-only physical candidate telemetry, not independently administered trust",
    }
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0 if all(item["status"] == "READ_ONLY_CANDIDATE"
                    for item in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
