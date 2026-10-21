"""Offline tests for the read-only Joyner duplicate-publisher witness."""
import importlib.util
import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    "audit_r9700_report_collision",
    Path(__file__).resolve().parents[1] / "scripts" / "audit_r9700_report_collision.py",
)
assert spec is not None and spec.loader is not None
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def example(*, populated=True, rich=False, received_at=1000):
    base = {
        "hostname": "joyner",
        "ip": "100.83.215.83",
        "received_at": received_at,
        "library": [{"id": "joyner-toolcall", "path": "/SECRET_MODEL_FILE"}] if populated else [],
        "loaded": [{"id": "spoofed-gpu-model"}],
        "specs": {"cpu_model": "Ryzen", "system_ram_gib": 14.75, "platform": "linux"},
        "power_profile": "daily",
        "power_model_class": "full",
        "secret": "do-not-print",
    }
    if rich:
        base["specs"] = {"gpu_backend": "vulkan", "gpu_device": "SECRET_GPU_PATH",
                         "gpu_vram_gib": 32, "system_ram_gib": 14.75}
        base["power_profile"] = ""
        base["power_model_class"] = ""
    return base


class CollisionWitnessTests(unittest.TestCase):
    def test_collision_of_two_formats_and_two_catalogs(self):
        values = [
            audit.summarize(example(populated=True, received_at=1000), now=1010),
            audit.summarize(example(populated=False, rich=True, received_at=1005), now=1010),
            audit.summarize(example(populated=True, received_at=1009), now=1010),
        ]
        got = audit.classify(values)
        self.assertEqual(got["finding"], "CONFLICTING_REPORT_FORMATS")
        self.assertEqual(got["report_formats"], 2)
        self.assertEqual(got["catalog_variants"], 2)
        self.assertFalse(got["source_pid_identified"])
        self.assertFalse(got["model_gpu_residency_proven"])
        self.assertEqual(got["production_admission"], "NO_GO")

    def test_single_format_switch_does_not_claim_multiple_publishers(self):
        a = audit.summarize(example(populated=True, received_at=1000), now=1010)
        b = audit.summarize(example(populated=False, received_at=1005), now=1010)
        self.assertEqual(audit.classify([a, b])["finding"], "ONE_FORMAT_OBSERVED")

    def test_two_formats_same_catalog_is_not_divergent(self):
        a = audit.summarize(example(populated=True, received_at=1000), now=1010)
        b = audit.summarize(example(populated=True, rich=True, received_at=1005), now=1010)
        self.assertEqual(audit.classify([a, b])["finding"],
                         "MULTIPLE_FORMATS_SAME_CATALOG")

    def test_reject_stale_evidence(self):
        a = audit.summarize(example(populated=True, received_at=100), now=1000)
        b = audit.summarize(example(populated=False, rich=True, received_at=100), now=1000)
        self.assertEqual(audit.classify([a, b])["finding"], "INSUFFICIENT_WITNESS")

    def test_unknown_host_not_attributed_to_joyner(self):
        d = example()
        d["hostname"] = "other-host"
        result = audit.summarize(d, now=1010)
        self.assertEqual(result["status"], "UNKNOWN")
        self.assertEqual(result["library_ids"], [])

    def test_secrets_not_in_output(self):
        raw = example(populated=True, rich=True, received_at=1000)
        result = audit.summarize(raw, now=1010)
        as_json = json.dumps(result)
        self.assertNotIn("do-not-print", as_json)
        self.assertNotIn("SECRET_GPU_PATH", as_json)
        self.assertNotIn("SECRET_MODEL_FILE", as_json)
        self.assertNotIn("spoofed-gpu-model", as_json)
        self.assertEqual(result["loaded_model_attestation"], "NOT_ATTESTED")

    def test_unrecognized_library_model_not_exposed(self):
        raw = example(received_at=1000)
        raw["library"] = ["secret-token-value"]
        self.assertEqual(audit.summarize(raw, now=1010)["library_ids"], [])

    def test_timestamp_outside_future_range_rejected(self):
        self.assertIsNone(audit.summarize(example(received_at=1000), now=900)["received_at"])

    def test_non_numeric_timestamp_rejected(self):
        x = example(received_at=1000)
        x["received_at"] = "1000"
        self.assertIsNone(audit.summarize(x, now=1010)["received_at"])

    def test_single_sample_proves_nothing(self):
        item = audit.summarize(example(received_at=1000), now=1010)
        self.assertEqual(audit.classify([item])["finding"], "INSUFFICIENT_WITNESS")

    def test_does_not_redirect_or_send_post(self):
        class FakeResponse:
            status = 302
            def read(self, *args):
                raise AssertionError("Redirect body must not be read")
        class FakeConnection:
            def __init__(self):
                self.requests = []
                self.closed = False
            def request(self, method, path, headers):
                self.requests.append((method, path))
            def getresponse(self):
                return FakeResponse()
            def close(self):
                self.closed = True
        conn = FakeConnection()
        with patch.object(audit.http.client, "HTTPConnection", return_value=conn) as make:
            self.assertIsNone(audit.get_report())
        self.assertEqual(conn.requests, [("GET", "/api/fleet/node/joyner")])
        self.assertTrue(conn.closed)
        self.assertEqual(make.call_args.args,
                         ("100.64.43.123", 8088))

    def test_bounded_body(self):
        class FakeResponse:
            status = 200
            def read(self, n):
                return b"{" + b"X" * n
        class FakeConnection:
            def request(self, *args, **kwargs):
                return None
            def getresponse(self):
                return FakeResponse()
            def close(self):
                return None
        with patch.object(audit.http.client, "HTTPConnection", return_value=FakeConnection()):
            self.assertIsNone(audit.get_report())

    def test_main_sends_zero_writes(self):
        record = example(received_at=1000)
        with patch.object(audit, "get_report", return_value=record), \
             patch.object(audit.time, "time", return_value=1010), \
             redirect_stdout(io.StringIO()) as output:
            code = audit.main(["--samples", "1", "--interval", "0"])
        self.assertEqual(code, 0)
        parsed = json.loads(output.getvalue())
        self.assertTrue(parsed["read_only"])
        self.assertEqual(parsed["assessment"]["finding"], "INSUFFICIENT_WITNESS")

    def test_rejects_unbounded_sample_count(self):
        with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as ex:
            audit.main(["--samples", "10000"])
        self.assertEqual(ex.exception.code, 2)


if __name__ == "__main__":
    unittest.main()