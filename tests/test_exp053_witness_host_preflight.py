"""Source-free mocks for the read-only physical witness preflight."""
import contextlib
import io
import json
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

RESEARCH = Path(__file__).resolve().parents[1] / "research"
sys.path.insert(0, str(RESEARCH))
import exp053_witness_host_preflight as probe

FIXED_CHALLENGE = "ab" * 16


def observation(**changes):
    result = {
        "challenge": FIXED_CHALLENGE,
        "python": "3.12.3",
        "sqlite": "3.45.1",
        "cryptography_available": True,
        "openssl_available": True,
        "tpm_device_present": False,
        "filesystem": "ext4",
        "free_bytes": 43_000_000_000,
        "clock_unix": 1792261000.0,
    }
    result.update(changes)
    return result


class SourceOnlyReadOnlyRemotePreflight(unittest.TestCase):
    def test_frozen_preflight_spec_and_scope(self):
        import hashlib
        self.assertEqual(hashlib.sha256(probe.SPEC.read_bytes()).hexdigest(),
                         probe.SPEC_SHA256)
        self.assertEqual(probe.BASE_DECISIONS["decision"], "NO_GO_RESEARCH_ONLY")
        self.assertIs(probe.BASE_DECISIONS["scientific_execution_authorized"], False)

    def test_pinned_ssh_no_agent_or_untrusted_hostkey(self):
        commands = []
        def runner(args, **kw):
            commands.append((args, kw))
            return subprocess.CompletedProcess(args, 0,
                                               json.dumps(observation()), "")
        with patch.object(probe.time, "time", return_value=1792261000.0):
            got = probe.probe(
                "lenovo", runner=runner, challenge_source=lambda n: FIXED_CHALLENGE)
        self.assertEqual(got["status"], "READ_ONLY_CANDIDATE")
        self.assertEqual(len(commands), 1)
        cmd, kwargs = commands[0]
        self.assertIn("StrictHostKeyChecking=yes", cmd)
        self.assertIn("ForwardAgent=no", cmd)
        self.assertIn("ClearAllForwardings=yes", cmd)
        self.assertIn("BatchMode=yes", cmd)
        self.assertEqual(cmd[-2], "lenovo")
        self.assertIn("python3 -c ", cmd[-1])
        self.assertNotIn("sudo", cmd[-1])
        self.assertNotIn("pip install", cmd[-1])
        self.assertEqual(kwargs["timeout"], 9)
        self.assertEqual(got["decision"], "NO_GO_RESEARCH_ONLY")
        self.assertFalse(got["independent_administration_approved"])

    def test_bad_aliases_denied_without_ssh(self):
        for host in ("-x", "lenovo;echo bad", "a/b", "", "x "):
            with self.subTest(alias=host):
                with self.assertRaises(ValueError):
                    probe.probe(host, runner=lambda *a, **k: self.fail("ran ssh"))

    def test_unsigned_remote_output_is_never_authorization(self):
        candidate = probe.parse_remote(json.dumps(observation()), FIXED_CHALLENGE)
        self.assertEqual(candidate["filesystem"], "ext4")
        self.assertNotIn("scientific_execution_authorized", candidate)
        self.assertNotIn("independent_administration_approved", candidate)

    def test_stale_challenge_denied(self):
        with self.assertRaisesRegex(ValueError, "stale"):
            probe.parse_remote(json.dumps(observation(challenge="cc"*16)),
                               FIXED_CHALLENGE)

    def test_extra_or_missing_sensitive_fields_denied(self):
        for change in ("extra_ip", "missing_tpm", "user"):
            result = observation()
            if change == "extra_ip":
                result["ip_address"] = "127.0.0.1"
            elif change == "missing_tpm":
                result.pop("tpm_device_present")
            else:
                result["ssh_user"] = "secret"
            with self.subTest(change=change), self.assertRaises(ValueError):
                probe.parse_remote(json.dumps(result), FIXED_CHALLENGE)

    def test_type_mismatch_or_nonfinite_denied(self):
        for key, val in (
            ("cryptography_available", "yes"), ("free_bytes", True),
            ("clock_unix", "1792261000"), ("filesystem", "/dev/sdb2"),
            ("sqlite", "unsupported"), ("free_bytes", -1),
        ):
            with self.subTest(field=key), self.assertRaises(ValueError):
                probe.parse_remote(json.dumps(observation(**{key: val})),
                                   FIXED_CHALLENGE)
        for snippet in ('{"x":NaN}', '{"x":Infinity}',
                        '{"challenge":"aa","challenge":"bb"}'):
            with self.assertRaises(ValueError):
                probe.parse_remote(snippet, FIXED_CHALLENGE)

    def test_ssh_denials_and_timeout_not_dropped(self):
        for condition in ("hostkey", "timeout", "malformed"):
            def runner(args, **kwargs):
                if condition == "timeout":
                    raise subprocess.TimeoutExpired(args, 9)
                if condition == "hostkey":
                    return subprocess.CompletedProcess(
                        args, 255, "", "HOST KEY VERIFICATION FAILED")
                return subprocess.CompletedProcess(args, 0,
                                                   '{"bad":"data"}', "")
            result = probe.probe(
                "lenovo", runner=runner, challenge_source=lambda n: FIXED_CHALLENGE)
            self.assertEqual(result["status"], "FAIL_CLOSED")
            self.assertNotIn("HOST KEY", json.dumps(result))
            self.assertFalse(result["operator_key_custody_approved"])

    def test_missing_capabilities_fail_read_only_qualification(self):
        for change in (
            {"cryptography_available": False},
            {"openssl_available": False},
            {"python": "3.8.0"},
            {"sqlite": "3.31.1"},
            {"free_bytes": 128 * 1024 ** 2},
        ):
            with self.subTest(change=change):
                def runner(args, **kw):
                    return subprocess.CompletedProcess(
                        args, 0, json.dumps(observation(**change)), "")
                with patch.object(probe.time, "time", return_value=1792261000.0):
                    item = probe.probe(
                        "lenovo", runner=runner,
                        challenge_source=lambda n: FIXED_CHALLENGE)
                self.assertEqual(item["status"], "CAPABILITY_HOLD")
                self.assertEqual(item["decision"], "NO_GO_RESEARCH_ONLY")

    def test_high_skew_never_authorizes(self):
        def runner(args, **kw):
            return subprocess.CompletedProcess(
                args, 0, json.dumps(observation(clock_unix=1792261050.0)), "")
        with patch.object(probe.time, "time", return_value=1792261000.0):
            result = probe.probe(
                "xwing", runner=runner, challenge_source=lambda n: FIXED_CHALLENGE)
        self.assertEqual(result["status"], "TIME_SKEW_HOLD")
        self.assertEqual(result["decision"], "NO_GO_RESEARCH_ONLY")

    def test_cli_preserves_individual_denial(self):
        def runner(args, **kw):
            if args[-2] == "destroyer":
                return subprocess.CompletedProcess(args, 255, "", "untrusted")
            # Parse challenge from decoded SSH remote Python source, not
            # from the shell-escaped command's quoting.
            import re
            import shlex
            source = shlex.split(args[-1])[-1]
            match = re.search(r"CHALLENGE_VALUE = '([0-9a-f]{32})'", source)
            self.assertIsNotNone(match)
            payload = observation(challenge=match.group(1))
            return subprocess.CompletedProcess(args, 0, json.dumps(payload), "")
        stdout = io.StringIO()
        with patch.object(probe.time, "time", return_value=1792261000.0):
            with contextlib.redirect_stdout(stdout):
                exit_code = probe.main(["lenovo", "destroyer"], runner=runner)
        report = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 2)
        self.assertEqual(len(report["hosts"]), 2)
        self.assertEqual(report["hosts"][1]["status"], "FAIL_CLOSED")
        self.assertEqual(report["decision"], "NO_GO_RESEARCH_ONLY")
        self.assertNotIn("untrusted", stdout.getvalue())

    def test_duplicate_aliases_refused(self):
        with self.assertRaises(SystemExit):
            probe.main(["lenovo", "lenovo"], runner=lambda *a, **k: None)


if __name__ == "__main__":
    unittest.main()
