"""Synthetic review-only UI/Python validation parity; no licensed RACE material."""

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UI = ROOT / "research" / "exp053_offline_review.html"
sys.path.insert(0, str(ROOT / "research"))
from exp053_race_review import check_decision


@unittest.skipUnless(shutil.which("node"), "Node is unavailable; JS parity test skipped")
class OfflineReviewValidationParity(unittest.TestCase):
    def test_js_completed_count_agrees_with_python_acceptance(self):
        article = ("In this entirely synthetic passage, a small observatory "
                   "measured rainfall using a sealed glass cylinder. "
                   "The described measurement did not rely on solar panels.")
        good_quote = "a small observatory measured rainfall"
        base = {
            "decision": "accept", "supported_option": "A", "foil_option": "B",
            "support_quote": good_quote,
            "foil_reason": "The alternative is not described in the synthetic passage.",
            "reject_reason": "",
        }
        fixtures = [("valid", {}), ("blank supported", {"supported_option": ""}),
                    ("blank foil", {"foil_option": ""}),
                    ("multi-letter supported", {"supported_option": "AB"}),
                    ("multi-letter foil", {"foil_option": "BC"}),
                    ("non-letter supported", {"supported_option": "X"}),
                    ("numeric supported", {"supported_option": 1}),
                    ("null foil", {"foil_option": None}),
                    ("identical answer and foil", {"foil_option": "A"}),
                    ("missing quote", {"support_quote": ""}),
                    ("non-string quote", {"support_quote": 123}),
                    ("quote not in article", {"support_quote": "That synthetic source never contains this span"}),
                    ("short foil rationale", {"foil_reason": "too short"}),
                    ("non-string foil rationale", {"foil_reason": ["not", "a", "string"]}),
                    ("unreviewed", {"decision": "unreviewed"}),
                    ("valid reject", {"decision": "reject", "reject_reason": "Ambiguous synthetic wording"}),
                    ("reject too short", {"decision": "reject", "reject_reason": "no"}),
                    ("reject non-string", {"decision": "reject", "reject_reason": 123}),
                    ("invalid decision", {"decision": "approved"})]
        checks = []
        for label, updates in fixtures:
            candidate = {**base, **updates}
            try:
                accepted = check_decision(candidate, {"article": article}) in {"accept", "reject"}
            except ValueError:
                accepted = False
            checks.append({"name": label, "decision": candidate, "python_accepts": accepted})
        self.assertTrue(checks[0]["python_accepts"])
        self.assertTrue(next(x for x in checks if x["name"] == "valid reject")["python_accepts"])
        for x in checks[1:]:
            if x["name"] != "valid reject":
                self.assertFalse(x["python_accepts"], x["name"])
        html = UI.read_text()
        js = html.split("<script>", 1)[1].split("</script>", 1)[0]
        with tempfile.TemporaryDirectory(prefix="exp053-offline-parity-synthetic-") as directory:
            base_path = Path(directory)
            (base_path / "inline.js").write_text(js)
            subprocess.run(["node", "--check", str(base_path / "inline.js")],
                           capture_output=True, text=True, check=True)
            mock_runner = r"""
const vm = require('vm'), fs = require('fs'), {webcrypto} = require('crypto');
const ctx = {document:{getElementById:()=>({addEventListener:()=>{}})},
  window:{addEventListener:()=>{}}, crypto:webcrypto, TextEncoder,console};
vm.createContext(ctx);
vm.runInContext(fs.readFileSync(process.argv[2],'utf8'),ctx);
const article=JSON.parse(process.argv[3]);
const cases=JSON.parse(process.argv[4]);
const answers=cases.map(item=>({name:item.name,
  js_complete:vm.runInContext('complete('+JSON.stringify(item.decision)+','+
                             JSON.stringify({article})+')',ctx)}));
console.log(JSON.stringify(answers));
"""
            (base_path / "runner.js").write_text(mock_runner)
            result = subprocess.run(
                ["node", str(base_path / "runner.js"), str(base_path / "inline.js"),
                 json.dumps(article), json.dumps(checks)], check=True,
                capture_output=True, text=True, timeout=9,
            )
        answers = json.loads(result.stdout)
        self.assertEqual(len(answers), len(checks))
        for answer, expect in zip(answers, checks):
            with self.subTest(case=expect["name"]):
                self.assertEqual(answer["name"], expect["name"])
                self.assertEqual(answer["js_complete"], expect["python_accepts"])

    def test_ui_still_cannot_connect_or_load_source_from_web(self):
        html = UI.read_text()
        self.assertIn("connect-src 'none'", html)
        self.assertNotIn("fetch(", html)
        self.assertNotIn("XMLHttpRequest", html)
        self.assertNotIn("localStorage", html)
        self.assertNotIn("sessionStorage", html)
        self.assertIn("Export unsigned review JSON", html)
        self.assertIn("no network", html.lower())


if __name__ == "__main__":
    unittest.main()
