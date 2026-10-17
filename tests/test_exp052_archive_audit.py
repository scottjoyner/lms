"""Source custody and corruption tests for EXP052 MiniCPM extension."""
import importlib.util
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
RESEARCH=Path(__file__).resolve().parents[1]/"research"
sys.path.insert(0,str(RESEARCH))
spec=importlib.util.spec_from_file_location("audit_exp052",RESEARCH/"audit_exp052.py")
exp=importlib.util.module_from_spec(spec);spec.loader.exec_module(exp)
class TestMiniCPMAudit(unittest.TestCase):
    def fixture(self):
        return json.loads((RESEARCH/"EXP052-MINICPM-RESULTS-2026-10-16.json").read_text())
    def test_locked_three_model_audit(self):
        b,m=exp.audit()
        self.assertEqual(sum(z["forwards"] for z in b.values())+m["forward_passes"],240)
    def test_wrong_revision_denied(self):
        m=self.fixture();m["hf_revision"]="changed"
        with self.assertRaisesRegex(ValueError,"HOLD"):exp.verify_mini(m)
    def test_wrong_corpus_count_denied(self):
        m=self.fixture();m["summary"]["strict"]["unrelated_swapped"]["correct"]=14
        with self.assertRaisesRegex(ValueError,"HOLD"):exp.verify_mini(m)
    def test_corrupt_source_digest_denied(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for name in exp.EXPECTED:
                shutil.copyfile(RESEARCH/name,root/name)
            src=root/"EXP052-PROTOCOL-2026-10-16.md"
            src.write_text(src.read_text()+"\n")
            with patch.object(exp,"ROOT",root):
                with self.assertRaisesRegex(ValueError,"HOLD"):
                    exp.audit()
    def test_report_tracks_limits(self):
        b,m=exp.audit()
        s=exp.report(b,m)
        self.assertIn("240 real",s)
        self.assertIn("cannot rank",s) if "cannot rank" in s else self.assertIn("No relative QA ability",s)
if __name__=="__main__":unittest.main()