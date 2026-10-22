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
from itertools import combinations

SCHEMA = "exp053-controller-ssh-custody-overlap-v1"
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


def evaluate(aliases, *, runner=subprocess.run, controller_user=None):
    if (not isinstance(aliases, (list, tuple)) or not 1 <= len(aliases) <= MAX_HOSTS or
            len(aliases) != len(set(aliases)) or
            any(not isinstance(a, str) or not HOST_RE.fullmatch(a) for a in aliases)):
        raise ValueError("invalid candidate aliases")
    if controller_user is None:
        controller_user = pwd.getpwuid(os.getuid()).pw_name
    observed = {a: inspect(a, runner=runner) for a in aliases}
    overlap = {a: False for a in aliases}
    for a, b in combinations(aliases, 2):
        if (observed[a] is not None and observed[b] is not None and
                observed[a]["identities"] & observed[b]["identities"]):
            overlap[a] = True
            overlap[b] = True
    hosts = []
    for alias in aliases:
        entry = observed[alias]
        hosts.append({
            "alias": alias,
            "status": "CONFIG_PARSED" if entry is not None else "FAIL_CLOSED",
            "same_controller_login": entry["user"] == controller_user if entry else None,
            "configured_identity_path_count": len(entry["identities"]) if entry else None,
            "configured_path_overlap_with_candidates": overlap[alias] if entry else None,
            "independent_administration_approved": False,
            "scientific_execution_authorized": False,
        })
    return {
        "schema": SCHEMA,
        "decision": DECISION,
        "scientific_execution_authorized": False,
        "independent_administration_approved": False,
        "configured_identity_path_overlap_detected": any(overlap.values()),
        "qualification_scope": "local SSH config path overlap; not actual key use, privileges or custody",
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
