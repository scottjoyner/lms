#!/usr/bin/env python3
"""Controller-only SSH credential-path overlap triage, never custody approval.

`ssh -G` reads the caller's existing SSH configuration without connecting to
remote hosts. WARNING: an SSH configuration with Match exec may execute a
locally configured command during evaluation; run only with trusted config.
"""
import argparse
import json
import os
import pwd
import re
import subprocess
import stat
from itertools import combinations

SCHEMA = "exp053-controller-ssh-custody-overlap-v2"
HOST_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,62}\Z")
MAX_STDOUT = 65536
MAX_HOSTS = 12
DECISION = "NO_GO_RESEARCH_ONLY"


def parse_effective_config(stdout):
    """Parse only narrowly needed SSH -G fields; never publish their values."""
    if not isinstance(stdout, str) or not stdout or len(stdout) > MAX_STDOUT or "\x00" in stdout:
        raise ValueError("invalid ssh configuration output")
    fields = {}
    identities = set()
    for line in stdout.splitlines():
        key, sep, value = line.partition(" ")
        # OpenSSH may emit mixed-case names (e.g. canonicalizePermittedcnames).
        if not sep or not value or not re.fullmatch(r"[A-Za-z][A-Za-z0-9]*", key):
            raise ValueError("invalid ssh config record")
        key = key.lower()
        if key == "identityfile":
            if value != "none":
                identities.add(value)
        elif key in ("user", "stricthostkeychecking", "batchmode"):
            if key in fields:
                raise ValueError("ambiguous ssh config record")
            fields[key] = value
    if not fields.get("user") or not identities:
        raise ValueError("missing SSH user or identity paths")
    if fields.get("stricthostkeychecking") not in ("yes", "true"):
        raise ValueError("host key check not strict")
    if fields.get("batchmode") != "yes":
        raise ValueError("SSH must be noninteractive")
    return {"user": fields["user"], "identities": identities}


def inspect(alias, *, runner=subprocess.run):
    if not isinstance(alias, str) or not HOST_RE.fullmatch(alias):
        raise ValueError("invalid explicit SSH alias")
    command = [
        "ssh", "-G", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
        "-o", "ControlMaster=no", "-o", "ControlPath=none",
        "-o", "UpdateHostKeys=no", "-o", "PermitLocalCommand=no",
        "-o", "ForwardAgent=no", "-o", "ClearAllForwardings=yes",
        alias,
    ]
    try:
        proc = runner(command, capture_output=True, text=True, timeout=6, check=False)
        if proc.returncode:
            raise ValueError("SSH config resolution failed")
        return parse_effective_config(proc.stdout)
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return None


def _local_file_identity(path):
    """Optional inode observation; no key bytes, contents or symlinks read.

    Unknown, missing, tokenized or nonregular paths cannot confirm identity.
    Never turn an absent/unreadable configured file into proof of isolation.
    """
    if not isinstance(path, str) or not path or any(c in path for c in ("%", "$", "\x00")):
        return None
    # Fleet Commander and similar orchestrators can override HOME with an
    # isolated sandbox. SSH's explicit ~/<path> is relative to the operating
    # system account home, not necessarily the process environment HOME.
    # Refuse ~another-user and unexpanded tokens rather than guessing.
    if path.startswith("~/"):
        expanded = os.path.join(pwd.getpwuid(os.getuid()).pw_dir, path[2:])
    elif path.startswith("~"):
        return None
    else:
        expanded = path
    if not os.path.isabs(expanded):
        return None
    try:
        item = os.lstat(expanded)
    except OSError:
        return None
    if not stat.S_ISREG(item.st_mode):
        return None
    return (item.st_dev, item.st_ino)


def _controller_agent_socket_present():
    """Only records whether an agent-like local socket exists, not key usage."""
    candidate = os.environ.get("SSH_AUTH_SOCK", "")
    if not candidate or not os.path.isabs(candidate):
        return False
    try:
        return stat.S_ISSOCK(os.stat(candidate).st_mode)
    except OSError:
        return False


def evaluate(aliases, *, runner=subprocess.run, controller_user=None,
             identity_probe=_local_file_identity, agent_socket_probe=_controller_agent_socket_present):
    if (not isinstance(aliases, (list, tuple)) or not 1 <= len(aliases) <= MAX_HOSTS or
            len(aliases) != len(set(aliases)) or
            any(not isinstance(a, str) or not HOST_RE.fullmatch(a) for a in aliases)):
        raise ValueError("invalid candidate aliases")
    if controller_user is None:
        controller_user = pwd.getpwuid(os.getuid()).pw_name
    observed = {a: inspect(a, runner=runner) for a in aliases}
    # Local stat evidence is supplemental. Remote authentication may still use
    # agent-held keys or other credential sources not observable through -G.
    local_files = {}
    local_stat_count = {}
    local_stat_incomplete = {}
    for alias, record in observed.items():
        identities = ([identity_probe(path) for path in record["identities"]]
                      if record is not None else [])
        local_files[alias] = {item for item in identities if item is not None}
        local_stat_count[alias] = sum(item is not None for item in identities)
        local_stat_incomplete[alias] = any(item is None for item in identities)
    inode_overlap = {a: False for a in aliases}
    overlap = {a: False for a in aliases}
    for a, b in combinations(aliases, 2):
        if (observed[a] is not None and observed[b] is not None and
                observed[a]["identities"] & observed[b]["identities"]):
            overlap[a] = True
            overlap[b] = True
        if (observed[a] is not None and observed[b] is not None and
                local_files[a] & local_files[b]):
            inode_overlap[a] = True
            inode_overlap[b] = True
    agent_socket_present = agent_socket_probe()
    hosts = []
    for alias in aliases:
        entry = observed[alias]
        hosts.append({
            "alias": alias,
            "status": "CONFIG_PARSED" if entry is not None else "FAIL_CLOSED",
            "same_controller_login": entry["user"] == controller_user if entry else None,
            "configured_identity_path_count": len(entry["identities"]) if entry else None,
            "configured_path_overlap_with_candidates": overlap[alias] if entry else None,
            "locally_verified_regular_identity_path_count": local_stat_count[alias] if entry else None,
            "local_identity_path_observation_incomplete": local_stat_incomplete[alias] if entry else None,
            "same_local_identity_file_inode_with_candidates": (
                inode_overlap[alias] if entry and local_files[alias] else None),
            "independent_administration_approved": False,
            "scientific_execution_authorized": False,
        })
    return {
        "schema": SCHEMA,
        "decision": DECISION,
        "scientific_execution_authorized": False,
        "independent_administration_approved": False,
        "configured_identity_path_overlap_detected": any(overlap.values()),
        "same_local_identity_file_inode_detected": any(inode_overlap.values()),
        "controller_ssh_agent_socket_present": bool(agent_socket_present),
        "qualification_scope": (
            "controller-only configured paths, local file metadata and agent socket presence; "
            "not authenticated key use, privileges or independent custody"),
        "hosts": hosts,
    }


def main(argv=None, *, runner=subprocess.run):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("aliases", nargs="+", help="Explicit known SSH aliases")
    args = parser.parse_args(argv)
    try:
        result = evaluate(args.aliases, runner=runner)
    except ValueError as exc:
        raise SystemExit("HOLD: " + str(exc)) from exc
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    # Deliberately never return a success signal usable for scientific admission.
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
