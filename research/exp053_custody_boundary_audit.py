#!/usr/bin/env python3
"""Source-free EXP-053 custody authority gap triage; ALWAYS NO-GO.

This program validates bounded *claims*, not proof of independent custody.
It never verifies human identities/signatures or changes approved trust pins.
"""

import argparse
import hashlib
import json
import os
import re
import stat
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SPEC = ROOT / "EXP053-CUSTODY-BOUNDARY-NEGATIVE-AUDIT-SPEC-2026-10-18.md"
SPEC_SHA256 = "46c06eab83811ea6028dab262a6bd1e7b127982b7ef70c17084938d29314b54a"
SCHEMA = "exp053-custody-boundary-claims-v1"
REPORT_SCHEMA = "exp053-custody-boundary-negative-audit-v1"
MAX_BYTES = 16 * 1024
MAX_CLAIMS = 24
DOMAINS = (
    "witness_remote_login",
    "witness_root_administration",
    "witness_signing_key_custody",
    "witness_highwater_mutation",
    "witness_snapshot_restore",
    "witness_backup_restore",
    "witness_trust_anchor_rotation",
    "witness_checkpoint_publication",
)
PRINCIPALS = ("scorer_operator", "fleet_automation", "witness_custodian")
ACCESS = {"CAN", "CANNOT", "UNKNOWN"}
ORIGINS = {"operator_observation", "custodian_claim", "synthetic_fixture", "none"}
CLAIM_FIELDS = {"domain", "principal", "access", "evidence_sha256", "evidence_origin"}
ALIAS_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,62}\Z")
SHA_RE = re.compile(r"[0-9a-f]{64}\Z")


class Hold(ValueError):
    """Safety HOLD; error strings never contain supplied JSON values."""


def _duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise Hold("duplicate JSON field")
        result[key] = value
    return result


def _reject_nonfinite(_value):
    raise Hold("nonfinite JSON value")


def _frozen_spec():
    if (SPEC.is_symlink() or not SPEC.is_file() or
            hashlib.sha256(SPEC.read_bytes()).hexdigest() != SPEC_SHA256):
        raise Hold("frozen prospective audit spec changed")


def parse(raw):
    _frozen_spec()
    if not isinstance(raw, bytes) or not 1 <= len(raw) <= MAX_BYTES:
        raise Hold("audit input is absent or outside byte limit")
    try:
        manifest = json.loads(raw.decode("utf-8", errors="strict"),
                              object_pairs_hook=_duplicates,
                              parse_constant=_reject_nonfinite)
    except (UnicodeError, ValueError, TypeError, RecursionError) as exc:
        # Very deep but still bounded JSON must remain a sanitized NO-GO,
        # not an unhandled parser exception with an internal traceback.
        raise Hold("malformed, ambiguous or deeply nested UTF-8 JSON") from exc
    if (type(manifest) is not dict or
            set(manifest) != {"schema", "candidate_alias", "claims"} or
            manifest["schema"] != SCHEMA or
            not isinstance(manifest["candidate_alias"], str) or
            not ALIAS_RE.fullmatch(manifest["candidate_alias"])):
        raise Hold("unrecognized custody manifest or alias")
    claims = manifest["claims"]
    if type(claims) is not list or not 1 <= len(claims) <= MAX_CLAIMS:
        raise Hold("missing or excessive custody claims")
    seen = set()
    for item in claims:
        if not isinstance(item, dict) or set(item) != CLAIM_FIELDS:
            raise Hold("incorrect custody claim fields")
        if (item["domain"] not in DOMAINS or
                item["principal"] not in PRINCIPALS or
                item["access"] not in ACCESS or
                item["evidence_origin"] not in ORIGINS):
            raise Hold("unrecognized custody scope or classification")
        key = (item["domain"], item["principal"])
        if key in seen:
            raise Hold("duplicate custody scope")
        seen.add(key)
        digest = item["evidence_sha256"]
        if digest is None:
            if item["evidence_origin"] != "none" or item["access"] != "UNKNOWN":
                raise Hold("unsubstantiated custody assertion")
        elif (not isinstance(digest, str) or not SHA_RE.fullmatch(digest) or
              item["evidence_origin"] == "none"):
            raise Hold("invalid custody evidence reference")
    return manifest


def assess(manifest):
    """No evidence verification: produces only negative/unknown classifications."""
    index = {(c["domain"], c["principal"]): c for c in manifest["claims"]}
    findings = []
    shared = []
    missing = 0
    for domain in DOMAINS:
        positions = [index.get((domain, principal)) for principal in PRINCIPALS]
        missing += sum(c is None for c in positions)
        disallowed = positions[:2]
        if any(c is not None and c["access"] == "CAN" for c in disallowed):
            verdict = "SHARED_CONTROL_CLAIM"
            shared.append(domain)
        elif (all(c is not None and c["access"] == "CANNOT" for c in disallowed) and
              positions[2] is not None and positions[2]["access"] == "CAN"):
            verdict = "SEPARATION_CLAIM_UNVERIFIED"
        else:
            verdict = "INCOMPLETE_OR_UNKNOWN"
        findings.append({"domain": domain, "classification": verdict})
    return {
        "schema": REPORT_SCHEMA,
        "candidate_alias": manifest["candidate_alias"],
        "decision": "NO_GO_RESEARCH_ONLY",
        "independent_administration_approved": False,
        "witness_durability_approved": False,
        "scientific_execution_authorized": False,
        "input_claim_count": len(manifest["claims"]),
        "missing_role_domain_claim_count": missing,
        "shared_control_claim_count": len(shared),
        "shared_control_claim_domains": shared,
        "domain_findings": findings,
        "evidence_status": "UNVERIFIED_DIGEST_REFERENCES_ONLY",
    }


def _read_bounded_regular_file(path):
    """Same-descriptor, no-follow, bounded reading of source-free claims."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        fd = os.open(path, flags)
        with os.fdopen(fd, "rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise Hold("input is not a regular file")
            return stream.read(MAX_BYTES + 1)
    except OSError as exc:
        raise Hold("cannot safely open audit input") from exc


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path,
                        help="Local source-free custody claims JSON (no private evidence)")
    args = parser.parse_args(argv)
    try:
        report = assess(parse(_read_bounded_regular_file(args.input)))
    except (Hold, OSError, ValueError):
        # Keep filenames, usernames and raw input strictly out of error output.
        print('{"decision":"NO_GO_RESEARCH_ONLY","scientific_execution_authorized":false,'
              '"status":"INVALID_OR_UNVERIFIABLE_INPUT"}')
        return 2
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 2  # Cannot authorize under ANY supplied claims.


if __name__ == "__main__":
    raise SystemExit(main())
