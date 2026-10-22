"""Offline, source-only tests; SSH is always mocked."""
import contextlib
import io
import json
import subprocess
import sys
from pathlib import Path
from unittest import TestCase

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research"))
import exp053_ssh_custody_overlap as audit


def config(user="scott", key="/path/to/shared-secret-name", *, extra=""):
    return ("user " + user + "\n"
            "identityfile " + key + "\n"
            "batchmode yes\nstricthostkeychecking true\n" + extra)


def runner_for(mapping):
    def fake(command, **options):
        assert command[0:2] == ["ssh", "-G"]
        for expected in ("BatchMode=yes", "StrictHostKeyChecking=yes", "ControlMaster=no",
                         "ControlPath=none", "UpdateHostKeys=no", "PermitLocalCommand=no",
                         "ForwardAgent=no", "ClearAllForwardings=yes"):
            assert expected in command
        assert options["timeout"] == 6
        return subprocess.CompletedProcess(command, 0, mapping[command[-1]], "")
    return fake


class ControllerCredentialOverlapTests(TestCase):
    def test_actual_config_paths_never_exposed_and_shared_overlap_detected(self):
        data = {"lenovo": config(), "destroyer": config(), "xwing": config()}
        got = audit.evaluate(list(data), runner=runner_for(data), controller_user="scott")
        self.assertTrue(got["configured_identity_path_overlap_detected"])
        self.assertEqual([x["configured_path_overlap_with_candidates"] for x in got["hosts"]],
                         [True, True, True])
        self.assertTrue(all(h["same_controller_login"] for h in got["hosts"]))
        self.assertEqual(got["decision"], "NO_GO_RESEARCH_ONLY")
        self.assertFalse(got["scientific_execution_authorized"])
        self.assertNotIn("shared-secret-name", json.dumps(got))
        self.assertNotIn("/path/", json.dumps(got))
        self.assertNotIn('"scott"', json.dumps(got))

    def test_distinct_paths_still_cannot_approve_independent_custody(self):
        data = {"lenovo": config(key="/path/one"),
                "destroyer": config(key="/path/two", user="other")}
        got = audit.evaluate(list(data), runner=runner_for(data), controller_user="scott")
        self.assertFalse(got["configured_identity_path_overlap_detected"])
        self.assertFalse(any(x["configured_path_overlap_with_candidates"] for x in got["hosts"]))
        self.assertFalse(got["independent_administration_approved"])
        self.assertFalse(got["scientific_execution_authorized"])

    def test_open_ssh_mixed_case_unknown_config_records_are_ignored(self):
        raw = config(extra="canonicalizePermittedcnames hidden-value\n")
        parsed = audit.parse_effective_config(raw)
        self.assertEqual(parsed["user"], "scott")
        self.assertEqual(len(parsed["identities"]), 1)

    def test_failed_host_not_dropped_or_misclassified(self):
        def fake(command, **_):
            if command[-1] == "destroyer":
                return subprocess.CompletedProcess(command, 255, "", "PRIVATE_IP_192.168.2.100")
            return subprocess.CompletedProcess(command, 0, config(), "")
        got = audit.evaluate(["lenovo", "destroyer"], runner=fake, controller_user="scott")
        self.assertEqual(len(got["hosts"]), 2)
        self.assertEqual(got["hosts"][1]["status"], "FAIL_CLOSED")
        self.assertIsNone(got["hosts"][1]["configured_path_overlap_with_candidates"])
        self.assertNotIn("192.168", json.dumps(got))
        self.assertFalse(got["scientific_execution_authorized"])

    def test_injected_ssh_alias_rejected_before_shell(self):
        for aliases in (["-x"], ["lenovo;id"], ["lenovo", "lenovo"], [],
                        ["lenovo", 8], ["a"] * 13):
            with self.subTest(aliases=aliases), self.assertRaises(ValueError):
                audit.evaluate(aliases, runner=lambda *a, **k: self.fail("ssh -G invoked"))

    def test_strict_output_parser_rejects_malformed_and_ambiguous(self):
        for raw in ("", "garbage", "identityfile /some/key\n", config(extra="user hijack\n"),
                    config(extra="stricthostkeychecking no\n"),
                    config().replace("true", "ask"), config().replace("batchmode yes", "batchmode no"),
                    config() + "\x00", config() + "x" * 65536, config(key="none")):
            with self.subTest(raw_len=len(raw)), self.assertRaises(ValueError):
                audit.parse_effective_config(raw)

    def test_cli_is_no_go_even_on_config_success(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = audit.main(["lenovo"], runner=runner_for({"lenovo": config()}))
        self.assertEqual(code, 2)
        data = json.loads(output.getvalue())
        self.assertEqual(data["decision"], "NO_GO_RESEARCH_ONLY")
        self.assertFalse(data["scientific_execution_authorized"])


if __name__ == "__main__":
    import unittest
    unittest.main()
