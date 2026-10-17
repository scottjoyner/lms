"""Source-free per-cell retry ledger software tests; NO native model runs."""
import copy
import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

RESEARCH = Path(__file__).resolve().parents[1] / "research"
sys.path.insert(0, str(RESEARCH))
import exp053_attempt_ledger as ledger
from exp053_race_review import canonical


def h(value):
    return hashlib.sha256(value).hexdigest()


class SyntheticAttemptLedger(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.private = Path(self.temp.name)
        self.logfile = self.private / "source-free-attempts.json"
        self.execution_id = "synthetic_case_test_only_001"
        self.packet = "a" * 64
        self.scores = [
            {"model": "K2-Horizon-3.7B", "case_id": "fake_case_001",
             "arm": "supported", "rotation": 0, "mode": "raw",
             "gold_logp": -1.0, "foil_logp": -2.0},
            {"model": "K2-Horizon-3.7B", "case_id": "fake_case_001",
             "arm": "decoy", "rotation": 1, "mode": "raw",
             "gold_logp": -1.4, "foil_logp": -2.5},
        ]
        entries = []
        for i, score in enumerate(self.scores):
            entry = {k: score[k] for k in ("model", "case_id", "arm", "rotation")}
            entry["score_row_sha256"] = h(canonical(score))
            entry["attempts"] = (
                [{"sequence": 1, "status": "failure", "reason": "timeout"},
                 {"sequence": 2, "status": "success", "reason": "none"}]
                if i == 0 else
                [{"sequence": 1, "status": "success", "reason": "none"}]
            )
            entries.append(entry)
        self.log = {"schema": ledger.SCHEMA,
                    "execution_id": self.execution_id,
                    "packet_sha256": self.packet,
                    "entries": entries}

    def save(self, raw=None):
        if raw is None:
            raw = json.dumps(self.log,sort_keys=True).encode()+b"\n"
        self.logfile.write_bytes(raw)
        os.chmod(self.logfile, 0o600)

    def verify(self, scores=None):
        self.save() if not self.logfile.exists() else None
        return ledger.verify_attempt_ledger(
            self.logfile, self.private,
            self.scores if scores is None else scores,
            execution_id_sha256=h(self.execution_id.encode()),
            packet_sha256=self.packet,
            attempt_log_sha256=h(self.logfile.read_bytes()))

    def reject(self):
        self.save()
        with self.assertRaisesRegex(ValueError, "HOLD"):
            self.verify()

    def test_frozen_source_only_protocol(self):
        self.assertEqual(ledger.filehash(ledger.SPEC), ledger.SPEC_SHA256)

    def test_complete_synthetic_attempts_accepted_as_shape_only(self):
        output = self.verify()
        self.assertEqual(output["attempt_cells"], 2)
        self.assertEqual(output["total_logged_attempts"], 3)
        self.assertEqual(output["logged_failed_attempts"], 1)
        self.assertNotIn("case_id", output)

    def test_missing_and_duplicate_cells_denied(self):
        for operation in ("missing","duplicate"):
            with self.subTest(operation=operation):
                old = copy.deepcopy(self.log)
                if operation == "missing": self.log["entries"].pop()
                else: self.log["entries"][1] = copy.deepcopy(self.log["entries"][0])
                self.reject()
                self.log = old

    def test_unknown_extra_entry_denied(self):
        self.log["entries"][0]["source_article"] = "private source forbidden"
        self.reject()

    def test_wrong_score_digest_denied(self):
        self.log["entries"][0]["score_row_sha256"] = "f" * 64
        self.reject()

    def test_score_row_mutation_denied(self):
        self.save()
        new_rows = copy.deepcopy(self.scores)
        new_rows[0]["gold_logp"] = -5.7
        with self.assertRaisesRegex(ValueError, "HOLD"):
            self.verify(new_rows)

    def test_missing_terminal_success_denied(self):
        self.log["entries"][0]["attempts"][-1] = {
            "sequence": 2, "status": "failure", "reason": "timeout"
        }
        self.reject()

    def test_early_success_then_failure_denied(self):
        self.log["entries"][0]["attempts"][0] = {
            "sequence": 1, "status": "success", "reason": "none"
        }
        self.reject()

    def test_non_contiguous_or_boolean_sequence_denied(self):
        for bad in (3, 0, True, 1.0):
            with self.subTest(bad=bad):
                self.log["entries"][0]["attempts"][0]["sequence"] = bad
                self.reject()

    def test_more_than_four_attempts_denied(self):
        self.log["entries"][0]["attempts"] = [
            {"sequence":i,"status":"failure" if i<5 else "success",
             "reason":"timeout" if i<5 else "none"} for i in range(1,6)
        ]
        self.reject()

    def test_invalid_failure_and_success_reasons_denied(self):
        for index,value in ((0,"none"),(0,"external_trust"),(1,"timeout")):
            with self.subTest(index=index,value=value):
                previous = self.log["entries"][0]["attempts"][index]["reason"]
                self.log["entries"][0]["attempts"][index]["reason"] = value
                self.reject()
                self.log["entries"][0]["attempts"][index]["reason"] = previous

    def test_wrong_execution_and_packet_denied(self):
        for field,bad in (("execution_id","synthetic_different_run"),
                          ("packet_sha256","b"*64)):
            previous = self.log[field]
            self.log[field] = bad
            self.reject()
            self.log[field] = previous

    def test_unexpected_fields_rejected(self):
        self.log["score_text"] = "no private text allowed"
        self.reject()

    def test_duplicate_key_and_nan_denied_even_when_digest_matches(self):
        self.save(b'{"schema":"v1","schema":"v1","entries":[]}\n')
        with self.assertRaisesRegex(ValueError,"HOLD"):
            self.verify()
        self.save(b'{"schema":"v1","entries":[NaN]}\n')
        with self.assertRaisesRegex(ValueError,"HOLD"):
            self.verify()

    def test_fingerprint_mismatch_denied(self):
        self.save()
        with self.assertRaisesRegex(ValueError,"HOLD"):
            ledger.verify_attempt_ledger(
                self.logfile,self.private,self.scores,
                execution_id_sha256=h(self.execution_id.encode()),
                packet_sha256=self.packet,attempt_log_sha256="c"*64)

    def test_symlinked_private_log_denied(self):
        self.save()
        duplicate = self.private / "real-ledger.json"
        duplicate.write_bytes(self.logfile.read_bytes())
        os.chmod(duplicate, 0o600)
        self.logfile.unlink()
        self.logfile.symlink_to(duplicate)
        with self.assertRaisesRegex(ValueError,"HOLD"):
            self.verify()

    def test_limit_applies_to_logged_payload(self):
        self.save()
        with patch.object(ledger,"MAX_LEDGER_BYTES",128):
            with self.assertRaisesRegex(ValueError,"HOLD"):
                self.verify()

    def test_bad_score_cell_types_denied(self):
        for field,bad in (("model",[]),("case_id",None),
                          ("rotation",True),("arm","imagined")):
            new_rows = copy.deepcopy(self.scores)
            new_rows[0][field] = bad
            self.save()
            with self.assertRaisesRegex(ValueError,"HOLD"):
                self.verify(new_rows)

if __name__ == "__main__":
    unittest.main()
