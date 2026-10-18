"""Synthetic, source-free test-plan generation; NO commands or witness writes."""
import contextlib
import hashlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

RESEARCH = Path(__file__).resolve().parents[1] / "research"
sys.path.insert(0, str(RESEARCH))
import exp053_custody_boundary_audit as controls
import exp053_custody_negative_challenges as planner


def claim(domain, principal, access="UNKNOWN", proof=None):
    return {"domain": domain, "principal": principal, "access": access,
            "evidence_sha256": (proof if access != "UNKNOWN" else None),
            "evidence_origin": ("synthetic_fixture" if access != "UNKNOWN" else "none")}


def document(claims, alias="lenovo"):
    return json.dumps({"schema": controls.SCHEMA, "candidate_alias": alias,
                       "claims": claims}, sort_keys=True, separators=(",", ":")).encode()


class ResearchOnlyPlan(unittest.TestCase):
    def test_frozen_challenges_spec_and_original_spec(self):
        self.assertEqual(hashlib.sha256(planner.SPEC.read_bytes()).hexdigest(),
                         planner.SPEC_SHA256)
        self.assertEqual(hashlib.sha256(controls.SPEC.read_bytes()).hexdigest(),
                         controls.SPEC_SHA256)

    def test_live_source_free_lenovo_claim_fixture_remains_unexecuted(self):
        source = (RESEARCH / "EXP053-LENOVO-CUSTODY-CONTROL-CLAIMS-2026-10-18.json").read_bytes()
        data = planner.plan(source)
        self.assertEqual(data["scenario_count"], 12)
        self.assertEqual(len(data["scenario_rows"]), 12)
        self.assertEqual(data["input_sha256"], hashlib.sha256(source).hexdigest())
        self.assertEqual(data["decision"], "NO_GO_RESEARCH_ONLY")
        self.assertFalse(data["all_scenarios_executed"])
        self.assertFalse(data["scientific_execution_authorized"])
        self.assertFalse(data["independent_administration_approved"])
        self.assertFalse(data["witness_durability_approved"])
        self.assertTrue(all(r["execution_status"] == "NOT_EXECUTED" for r in data["scenario_rows"]))
        self.assertTrue(all(r["evidence_status"] == "NOT_ATTESTED" for r in data["scenario_rows"]))
        self.assertEqual(data["scenario_rows"][0]["claim_status"], "CLAIMED_CAN")
        self.assertEqual(data["scenario_rows"][1]["claim_status"], "MISSING")
        self.assertEqual(data["scenario_rows"][2]["claim_status"], "CLAIMED_UNKNOWN")

    def test_all_synthetic_favorable_claims_still_unexecuted(self):
        entries = []
        for domain in controls.DOMAINS:
            entries.extend([claim(domain, "scorer_operator", "CANNOT", "b" * 64),
                            claim(domain, "fleet_automation", "CANNOT", "b" * 64),
                            claim(domain, "witness_custodian", "CAN", "b" * 64)])
        data = planner.plan(document(entries))
        self.assertTrue(all(r["claim_status"] == "CLAIMED_CANNOT_UNVERIFIED"
                            for r in data["scenario_rows"]))
        self.assertFalse(data["all_scenarios_executed"])
        self.assertFalse(data["independent_administration_approved"])
        self.assertFalse(data["scientific_execution_authorized"])

    def test_all_scorer_or_automation_can_still_is_only_unexecuted_plan(self):
        entries = [claim(d, p, "CAN", "c" * 64) for d in controls.DOMAINS
                   for p in controls.PRINCIPALS]
        data = planner.plan(document(entries))
        self.assertTrue(all(r["claim_status"] == "CLAIMED_CAN" for r in data["scenario_rows"]))
        self.assertTrue(all(r["execution_status"] == "NOT_EXECUTED" for r in data["scenario_rows"]))
        self.assertEqual(data["decision"], "NO_GO_RESEARCH_ONLY")

    def test_expected_scenarios_are_fixed_source_free_and_complete(self):
        self.assertEqual(len(planner.SCENARIOS), 12)
        self.assertEqual(len(set(x[0] for x in planner.SCENARIOS)), 12)
        self.assertTrue({x[1] for x in planner.SCENARIOS} == set(controls.DOMAINS))
        for row in planner.SCENARIOS:
            self.assertIn(row[2], ("scorer_operator", "fleet_automation"))
            self.assertIn(row[1], controls.DOMAINS)
            self.assertNotIn("ssh -", row[3])
            self.assertNotIn("sudo -", row[3])

    def test_plan_never_echoes_private_digest_or_evidence_origin(self):
        secret = "0123456789abcdef" * 4
        raw = document([claim("witness_highwater_mutation", "scorer_operator",
                              "CAN", secret)])
        output = json.dumps(planner.plan(raw))
        self.assertNotIn(secret, output)
        self.assertNotIn("synthetic_fixture", output)
        self.assertNotIn("evidence_sha256", output)
        self.assertNotIn("operator_observation", output)

    def test_repeated_calls_byte_deterministic_without_clock_or_randomness(self):
        raw = document([claim("witness_remote_login", "scorer_operator")])
        a = json.dumps(planner.plan(raw), sort_keys=True)
        b = json.dumps(planner.plan(raw), sort_keys=True)
        self.assertEqual(a, b)

    def test_malformed_and_duplicate_inputs_fail_closed(self):
        for raw in (b"", b"{}", b"\xff", b'{"candidate_alias":"abc","candidate_alias":"def"}',
                    document([claim("witness_remote_login", "scorer_operator")]) + b"\x00"):
            with self.subTest(raw_len=len(raw)), self.assertRaises(controls.Hold):
                planner.plan(raw)

    def test_protocol_drift_refused(self):
        raw = document([claim("witness_remote_login", "scorer_operator")])
        with patch.object(planner, "SPEC_SHA256", "0" * 64):
            with self.assertRaisesRegex(controls.Hold, "frozen"):
                planner.plan(raw)
        with patch.object(controls, "SPEC_SHA256", "0" * 64):
            with self.assertRaisesRegex(controls.Hold, "frozen"):
                planner.plan(raw)

    def test_cli_exits_two_without_writing_or_exposing_input_path(self):
        with tempfile.TemporaryDirectory(prefix="exp053-custody-plan-") as tmp:
            root = Path(tmp)
            path = root / "private-name-source-free.json"
            path.write_bytes(document([claim("witness_remote_login", "scorer_operator")]))
            before = sorted(x.name for x in root.iterdir())
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                exitcode = planner.main(["--input", str(path)])
            self.assertEqual(before, sorted(x.name for x in root.iterdir()))
            self.assertEqual(exitcode, 2)
            self.assertEqual(json.loads(output.getvalue())["scenario_count"], 12)
            self.assertNotIn(str(root), output.getvalue())
            self.assertNotIn(path.name, output.getvalue())

    def test_link_and_fifo_refused_without_blocking(self):
        with tempfile.TemporaryDirectory(prefix="exp053-custody-plan-") as tmp:
            root = Path(tmp)
            original = root / "original.json"
            original.write_bytes(document([claim("witness_remote_login", "scorer_operator")]))
            link = root / "link.json"
            link.symlink_to(original)
            pipe = root / "pipe.json"
            os.mkfifo(pipe)
            for candidate in (link, pipe):
                with self.subTest(candidate=candidate.name):
                    output = io.StringIO()
                    with contextlib.redirect_stdout(output):
                        code = planner.main(["--input", str(candidate)])
                    self.assertEqual(code, 2)
                    self.assertEqual(json.loads(output.getvalue())["status"],
                                     "INVALID_OR_UNVERIFIABLE_INPUT")
                    self.assertNotIn(str(root), output.getvalue())


if __name__ == "__main__":
    unittest.main()
