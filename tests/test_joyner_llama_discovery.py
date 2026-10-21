import importlib.util
import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

# The packaged CI test runner does not put repository root on sys.path.
# Load the standalone reporter explicitly from its tracked path.
_SPEC = importlib.util.spec_from_file_location(
    "fleet_node_reporter", Path(__file__).resolve().parents[1] / "fleet_node_reporter.py"
)
assert _SPEC is not None and _SPEC.loader is not None
reporter = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(reporter)


class JoynerDiscoveryTests(unittest.TestCase):
    def _good(self, port, endpoint):
        alias, path = {
            value[0]: (value[1], value[2])
            for value in reporter._JOYNER_LANES
        }[port]
        return {
            "/health": {"status": "ok"},
            "/props": {"model_alias": alias, "model_path": path},
            "/v1/models": {"data": [{"id": alias}]},
        }[endpoint]

    def test_valid_two_lanes_catalog_only(self):
        with patch.object(reporter.socket, "gethostname", return_value="joyner"), \
             patch.object(reporter, "_joyner_loopback_json", side_effect=self._good):
            entries = reporter._joyner_llama_library()
        self.assertEqual(
            [entry["id"] for entry in entries], ["joyner-toolcall", "joyner-ling"]
        )
        self.assertEqual([entry["path"] for entry in entries], ["", ""])

    def test_not_other_hosts(self):
        with patch.object(reporter.socket, "gethostname", return_value="x1-370"), \
             patch.object(reporter, "_joyner_loopback_json") as probe:
            self.assertEqual(reporter._joyner_llama_library(), [])
            probe.assert_not_called()

    def test_health_bad_does_not_advertise(self):
        def bad(port, endpoint):
            return {"status": "unhealthy"} if endpoint == "/health" else self._good(port, endpoint)
        with patch.object(reporter.socket, "gethostname", return_value="joyner"), \
             patch.object(reporter, "_joyner_loopback_json", side_effect=bad):
            self.assertEqual(reporter._joyner_llama_library(), [])

    def test_changed_path_rejected(self):
        def bad(port, endpoint):
            return {"model_alias": self._good(port, "/props")["model_alias"],
                    "model_path": "/tmp/fake.gguf"} if endpoint == "/props" else self._good(port, endpoint)
        with patch.object(reporter.socket, "gethostname", return_value="joyner"), \
             patch.object(reporter, "_joyner_loopback_json", side_effect=bad):
            self.assertEqual(reporter._joyner_llama_library(), [])

    def test_wrong_alias_rejected(self):
        def bad(port, endpoint):
            return {"model_alias": "not-joyner", "model_path": self._good(port, "/props")["model_path"]} \
                   if endpoint == "/props" else self._good(port, endpoint)
        with patch.object(reporter.socket, "gethostname", return_value="joyner"), \
             patch.object(reporter, "_joyner_loopback_json", side_effect=bad):
            self.assertEqual(reporter._joyner_llama_library(), [])

    def test_catalog_missing_rejected(self):
        def bad(port, endpoint):
            return {"data": [{"id": "unrelated"}]} if endpoint == "/v1/models" else self._good(port, endpoint)
        with patch.object(reporter.socket, "gethostname", return_value="joyner"), \
             patch.object(reporter, "_joyner_loopback_json", side_effect=bad):
            self.assertEqual(reporter._joyner_llama_library(), [])

    def test_one_lane_down_only_other_discovered(self):
        def partial(port, endpoint):
            return None if port == 1236 else self._good(port, endpoint)
        with patch.object(reporter.socket, "gethostname", return_value="joyner"), \
             patch.object(reporter, "_joyner_loopback_json", side_effect=partial):
            self.assertEqual(
                [e["id"] for e in reporter._joyner_llama_library()], ["joyner-toolcall"]
            )

    def test_opt_in_default_off_preserves_existing_report(self):
        with patch.object(reporter, "_specs", return_value={"system_ram_gib": 14.7}), \
             patch.object(reporter.socket, "gethostname", return_value="joyner"), \
             patch.object(reporter, "_library", return_value=[]), \
             patch.object(reporter, "_loaded_models", return_value=[]), \
             patch.object(reporter, "_joyner_llama_library") as probe:
            result = reporter.build_report("http://127.0.0.1:1234")
        self.assertEqual(result["library"], [])
        self.assertEqual(result["loaded"], [])
        probe.assert_not_called()

    def test_opt_in_does_not_turn_catalog_into_loaded(self):
        with patch.object(reporter, "_specs", return_value={}), \
             patch.object(reporter.socket, "gethostname", return_value="joyner"), \
             patch.object(reporter, "_library", return_value=[]), \
             patch.object(reporter, "_loaded_models", return_value=[]), \
             patch.object(reporter, "_joyner_llama_library",
                          return_value=[{"id": "joyner-ling", "path": ""}]):
            result = reporter.build_report("http://127.0.0.1:1234",
                                           joyner_llama_discovery=True)
        self.assertEqual([x["id"] for x in result["library"]], ["joyner-ling"])
        self.assertEqual(result["loaded"], [])

    def test_existing_id_is_not_duplicated(self):
        with patch.object(reporter, "_specs", return_value={}), \
             patch.object(reporter.socket, "gethostname", return_value="joyner"), \
             patch.object(reporter, "_library",
                          return_value=[{"id":"joyner-toolcall","path":""}]), \
             patch.object(reporter, "_loaded_models", return_value=[]), \
             patch.object(reporter, "_joyner_llama_library",
                          return_value=[{"id":"joyner-toolcall","path":""},
                                        {"id":"joyner-ling","path":""}]):
            result = reporter.build_report("http://127.0.0.1:1234",
                                           joyner_llama_discovery=True)
        self.assertEqual(len(result["library"]), 2)

    def test_dry_run_never_posts(self):
        with patch.object(reporter, "_specs", return_value={}), \
             patch.object(reporter.socket, "gethostname", return_value="joyner"), \
             patch.object(reporter, "_library", return_value=[]), \
             patch.object(reporter, "_loaded_models", return_value=[]), \
             patch.object(reporter, "_joyner_llama_library", return_value=[]), \
             patch.object(reporter, "post_report") as post, \
             redirect_stdout(io.StringIO()) as buf:
            self.assertEqual(reporter.main(["--dry-run", "--joyner-llama-discovery"]), 0)
        self.assertEqual(json.loads(buf.getvalue())["loaded"], [])
        post.assert_not_called()

    def test_url_allowlist_never_accepts_unrecognized_port_or_route(self):
        with patch.object(reporter.http.client, "HTTPConnection") as conn:
            self.assertIsNone(reporter._joyner_loopback_json(1237, "/health"))
            self.assertIsNone(reporter._joyner_loopback_json(1234, "/v1/chat/completions"))
            conn.assert_not_called()

    def test_local_get_does_not_follow_redirect(self):
        class FakeResponse:
            status = 302
            def read(self, *args):
                raise AssertionError("must not read redirect body")
        class FakeConnection:
            def __init__(self):
                self.closed = False
                self.commands = []
            def request(self, method, endpoint, headers=None):
                self.commands.append((method, endpoint))
            def getresponse(self):
                return FakeResponse()
            def close(self):
                self.closed = True
        c = FakeConnection()
        with patch.object(reporter.http.client, "HTTPConnection", return_value=c) as mk:
            self.assertIsNone(reporter._joyner_loopback_json(1234, "/health"))
            self.assertEqual(c.commands, [("GET", "/health")])
            self.assertTrue(c.closed)
            self.assertEqual(mk.call_args.args, ("127.0.0.1", 1234))

    def test_oversize_payload_rejected(self):
        class FakeResponse:
            status=200
            def read(self, n):
                return b"{" + b" " * n
        class FakeConnection:
            def request(self, *args, **kwargs): pass
            def getresponse(self): return FakeResponse()
            def close(self): pass
        with patch.object(reporter.http.client, "HTTPConnection", return_value=FakeConnection()):
            self.assertIsNone(reporter._joyner_loopback_json(1234, "/health"))


if __name__ == "__main__":
    unittest.main()