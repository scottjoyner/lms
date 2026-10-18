"""Source-only negative custody audit tests. Never use real fleet/keys/models."""

import contextlib
import copy
import hashlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sys import path as syspath

RESEARCH = Path(__file__).resolve().parents[1] / "research"
syspath.insert(0, str(RESEARCH))
import exp053_custody_boundary_audit as audit


HASH = "a" * 64  # Non-authoritative synthetic digest, not a receipt.


def claim(domain, principal, access, *, evidence=HASH, origin="synthetic_fixture"):
    return {"domain": domain, "principal": principal, "access": access,
            "evidence_sha256": evidence, "evidence_origin": origin}


def corpus(claims=None):
    return {"schema": audit.SCHEMA, "candidate_alias": "lenovo",
            "claims": claims if claims is not None else [
                claim("witness_remote_login", "scorer_operator", "CAN")]}


def decode(value):
    return audit.parse(json.dumps(value, separators=(",", ":")).encode())


class CustodyBoundaryNegativeCases(unittest.TestCase):
    def test_prospective_protocol_is_frozen(self):
        self.assertEqual(hashlib.sha256(audit.SPEC.read_bytes()).hexdigest(),
                         audit.SPEC_SHA256)
        self.assertIn("NO_GO_RESEARCH_ONLY", audit.SPEC.read_text())

    def test_scoring_operator_access_explicitly_flagged_as_claim_only(self):
        report = audit.assess(decode(corpus()))
        self.assertEqual(report["shared_control_claim_domains"],
                         ["witness_remote_login"])
        self.assertEqual(report["shared_control_claim_count"], 1)
        self.assertEqual(report["missing_role_domain_claim_count"], 23)
        self.assertEqual(report["decision"], "NO_GO_RESEARCH_ONLY")
        self.assertFalse(report["scientific_execution_authorized"])
        self.assertFalse(report["independent_administration_approved"])
        self.assertFalse(report["witness_durability_approved"])
        self.assertNotIn(HASH, json.dumps(report))
        self.assertNotIn("synthetic_fixture", json.dumps(report))

    def test_all_claimed_separation_is_still_unverified(self):
        claims = []
        for domain in audit.DOMAINS:
            claims.extend((
                claim(domain, "scorer_operator", "CANNOT", origin="custodian_claim"),
                claim(domain, "fleet_automation", "CANNOT", origin="custodian_claim"),
                claim(domain, "witness_custodian", "CAN", origin="custodian_claim")))
        report = audit.assess(decode(corpus(claims)))
        self.assertEqual(report["input_claim_count"], 24)
        self.assertEqual(report["missing_role_domain_claim_count"], 0)
        self.assertEqual(report["shared_control_claim_count"], 0)
        self.assertTrue(all(x["classification"] == "SEPARATION_CLAIM_UNVERIFIED"
                            for x in report["domain_findings"]))
        self.assertFalse(report["independent_administration_approved"])
        self.assertFalse(report["scientific_execution_authorized"])
        self.assertEqual(report["evidence_status"], "UNVERIFIED_DIGEST_REFERENCES_ONLY")

    def test_unknown_and_missing_are_not_promoted_to_denied(self):
        value = corpus([
            claim("witness_snapshot_restore", "scorer_operator", "UNKNOWN",
                  evidence=None, origin="none"),
            claim("witness_snapshot_restore", "fleet_automation", "CANNOT"),
            claim("witness_snapshot_restore", "witness_custodian", "CAN")])
        report = audit.assess(decode(value))
        by_domain = {x["domain"]: x["classification"] for x in report["domain_findings"]}
        self.assertEqual(by_domain["witness_snapshot_restore"], "INCOMPLETE_OR_UNKNOWN")
        self.assertEqual(report["missing_role_domain_claim_count"], 21)

    def test_fail_closed_on_malformed_structures(self):
        source = corpus()
        cases = [
            b"", b"\xff", b"not-json", b"{\"x\":NaN}",
            b'{"schema":"a","schema":"b"}',
            json.dumps(source).encode() + b"x" * audit.MAX_BYTES,
            json.dumps({**source, "unexpected": "leak"}).encode(),
            json.dumps({**source, "candidate_alias": "host;sudo"}).encode(),
            json.dumps({**source, "claims": []}).encode(),
            json.dumps({**source, "claims": source["claims"] * 25}).encode(),
            json.dumps({**source, "claims": source["claims"] * 2}).encode(),
            json.dumps({**source, "claims": [{**source["claims"][0], "path": "/secret"}]}).encode(),
            json.dumps({**source, "claims": [{**source["claims"][0], "domain": []}]}).encode(),
            json.dumps({**source, "claims": [{**source["claims"][0], "access": "YES"}]}).encode(),
            json.dumps({**source, "claims": [{**source["claims"][0], "evidence_sha256": "Z" * 64}]}).encode(),
            json.dumps({**source, "claims": [{**source["claims"][0], "evidence_sha256": None}]}).encode(),
            json.dumps({**source, "claims": [{**source["claims"][0], "evidence_origin": "none"}]}).encode(),
        ]
        for data in cases:
            with self.subTest(size=len(data)), self.assertRaises(audit.Hold):
                audit.parse(data)

    def test_missing_source_custody_spec_disables_parser(self):
        with patch.object(audit, "SPEC_SHA256", "0" * 64):
            with self.assertRaisesRegex(audit.Hold, "frozen"):
                decode(corpus())

    def test_unknown_without_digest_is_valid_but_non_authoritative(self):
        row = claim("witness_backup_restore", "scorer_operator", "UNKNOWN",
                    evidence=None, origin="none")
        report = audit.assess(decode(corpus([row])))
        self.assertFalse(report["scientific_execution_authorized"])
        self.assertEqual(report["domain_findings"][5]["classification"],
                         "INCOMPLETE_OR_UNKNOWN")

    def test_cli_always_returns_two_and_never_prints_input_path(self):
        with tempfile.TemporaryDirectory(prefix="exp053-boundary-") as tmp:
            p = Path(tmp) / "sensitive-filename.json"
            p.write_text(json.dumps(corpus()))
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = audit.main(["--input", str(p)])
        self.assertEqual(code, 2)
        self.assertEqual(json.loads(out.getvalue())["decision"], "NO_GO_RESEARCH_ONLY")
        self.assertNotIn("sensitive-filename", out.getvalue())
        self.assertNotIn(tmp, out.getvalue())

    def test_symlink_and_special_file_fail_closed(self):
        with tempfile.TemporaryDirectory(prefix="exp053-boundary-") as tmp:
            root = Path(tmp)
            original = root / "source.json"
            original.write_text(json.dumps(corpus()))
            link = root / "alias.json"
            link.symlink_to(original)
            pipe = root / "pipe"
            os.mkfifo(pipe)
            for candidate in (link, pipe):
                with self.subTest(candidate=candidate.name):
                    out = io.StringIO()
                    with contextlib.redirect_stdout(out):
                        result = audit.main(["--input", str(candidate)])
                    self.assertEqual(result, 2)
                    self.assertEqual(json.loads(out.getvalue())["status"],
                                     "INVALID_OR_UNVERIFIABLE_INPUT")
                    self.assertNotIn(str(root), out.getvalue())

    def test_false_claims_can_never_turn_on_approval(self):
        item = corpus()
        for role in audit.PRINCIPALS:
            for status in ("CAN", "CANNOT", "UNKNOWN"):
                case = copy.deepcopy(item)
                case["claims"][0]["principal"] = role
                case["claims"][0]["access"] = status
                if status == "UNKNOWN":
                    case["claims"][0]["evidence_sha256"] = None
                    case["claims"][0]["evidence_origin"] = "none"
                report = audit.assess(decode(case))
                self.assertEqual(report["decision"], "NO_GO_RESEARCH_ONLY")
                self.assertFalse(report["independent_administration_approved"])
                self.assertFalse(report["witness_durability_approved"])
                self.assertFalse(report["scientific_execution_authorized"])


if __name__ == "__main__":
    unittest.main()
