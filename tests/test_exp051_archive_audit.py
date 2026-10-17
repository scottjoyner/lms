"""Fail-closed regression tests for source-locked EXP051 aggregate audit."""
import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1]/"research"/"audit_exp051.py"
spec = importlib.util.spec_from_file_location("audit_exp051",SOURCE)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)

class TestEXP051Archive(unittest.TestCase):
    def test_source_locked_summary(self):
        out = audit.audit()
        self.assertEqual(set(out),{"k2","qwen"})
        self.assertEqual(sum(x["forwards"] for x in out.values()),160)
        self.assertEqual(sum(z["correct"] for x in out.values()
                         for z in x["summary"]["strict"].values()),160)
    def test_rejects_corrupted_result(self):
        result = json.loads((audit.ROOT/"EXP051-K2-3P7B-RESULTS-2026-10-16.json").read_text())
        result["summary"]["strict"]["supported_original"]["correct"] = 15
        with self.assertRaisesRegex(ValueError,"HOLD"):
            audit.validate_result(result,"k2")
    def test_rejects_model_mislabel(self):
        result = json.loads((audit.ROOT/"EXP051-QWEN4B-RESULTS-2026-10-16.json").read_text())
        with self.assertRaisesRegex(ValueError,"HOLD"):
            audit.validate_result(result,"k2")
    def test_rejects_changed_source_digest(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for name in audit.INPUTS:
                shutil.copyfile(audit.ROOT/name,root/name)
            audit.check_files(root)
            changed=root/"EXP051-QWEN4B-RESULTS-2026-10-16.json"
            changed.write_text(changed.read_text()+"\n")
            with self.assertRaisesRegex(ValueError,"HOLD"):
                audit.check_files(root)
    def test_report_contains_both_models_and_limits(self):
        s=audit.report(audit.audit())
        self.assertIn("Qwen3.5-4B",s)
        self.assertIn("K2-Horizon-3.7B",s)
        self.assertIn("not 80 independent",s)
if __name__=="__main__":unittest.main()