"""EXP-041B one-shot resource guard and source-custody wrapper tests."""
import pathlib
import subprocess
import unittest
P=pathlib.Path(__file__).resolve().parents[1]/"research"
SH=P/"run_k2_exp041b_if_admitted.sh"

class LauncherTests(unittest.TestCase):
    def test_shell_syntax(self):
        done=subprocess.run(["bash","-n",str(SH)],capture_output=True,text=True)
        self.assertEqual(done.returncode,0,done.stderr)
    def test_restrictions_present(self):
        content=SH.read_text()
        self.assertIn("40 * 1024 * 1024",content)
        self.assertIn('flock -n 9',content)
        self.assertIn("timeout --signal=TERM --kill-after=10s 650s",content)
        self.assertIn("research/k2_new_source_generation_b.py",content)
        self.assertIn("EXPERIMENT-041B-RESULTS-2026-10-15.md",content)
        self.assertIn('[[ ! -e "$REPORT" ]]',content)
    def test_model_free_preflight(self):
        completed=subprocess.run(["bash",str(SH),"--preflight-only"],
            capture_output=True,text=True,timeout=12)
        self.assertEqual(completed.returncode,0,completed.stdout+completed.stderr)
        self.assertIn("EXP041B_PREFLIGHT_FIXTURE_OK",completed.stdout)
        self.assertIn("DRY_RUN_NO_MODEL_LOADING_OR_WRITES",completed.stdout)
if __name__=="__main__":
    unittest.main()
