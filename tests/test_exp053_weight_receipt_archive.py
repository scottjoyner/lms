"""Public source-free checkpoint hash receipt consistency, NOT human approval.

Checks the measured read-only inventory, not model execution or reviewer status.
"""
import copy
import hashlib
import json
import sys
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]/"research"
sys.path.insert(0,str(ROOT))
import exp053_weight_identity as w
import exp053_verified_analysis as analysis

REPORT=ROOT/"EXP053-WEIGHT-IDENTITY-2026-10-16.json"
ARCHIVE_SHA256="e95c771517869f2060b84c7a270bc3e59bd659500c5187d2227d2f94d00440b6"
EXPECTED_COUNTS={"K2-Horizon-3.7B":36,"Qwen3.5-4B-Base":2,"MiniCPM-V-4.7-1B":1}
EXPECTED_BYTES={"K2-Horizon-3.7B":10116554080,
                "Qwen3.5-4B-Base":9319828056,
                "MiniCPM-V-4.7-1B":3109054480}


class ArchiveReceipt(unittest.TestCase):
    def test_exact_source_free_receipt(self):
        self.assertTrue(REPORT.is_file())
        self.assertEqual(hashlib.sha256(REPORT.read_bytes()).hexdigest(),ARCHIVE_SHA256)
        d=json.loads(REPORT.read_text())
        self.assertEqual(set(d["models"]),set(EXPECTED_COUNTS))
        self.assertEqual(d["schema"],"exp053-source-only-weight-identity-v1")
        self.assertEqual(d["spec_sha256"],w.SPEC_SHA)
        self.assertEqual(d["tokenizer_receipt_sha256"],w.TOKENS_REPORT_SHA)
        self.assertEqual(d["neural_forward_passes"],0)
        self.assertEqual(d["human_reviewed_articles"],0)
        self.assertEqual(w.sha(d["models"]),d["checkpoint_receipts_sha256"])
        for model,m in d["models"].items():
            self.assertEqual(m["model"],model)
            self.assertEqual(m["shard_count"],EXPECTED_COUNTS[model])
            self.assertEqual(m["total_weight_bytes"],EXPECTED_BYTES[model])
            self.assertEqual(sum(x["bytes"] for x in m["shards"]),m["total_weight_bytes"])
            self.assertEqual(m["shards"],sorted(m["shards"],key=lambda x:x["filename"]))
            self.assertEqual(len({x["filename"] for x in m["shards"]}),m["shard_count"])
            self.assertTrue(all(len(x["sha256"])==64 for x in m["shards"]))
            inner=copy.deepcopy(m)
            inner.pop("checkpoint_receipt_sha256")
            self.assertEqual(w.sha(inner),m["checkpoint_receipt_sha256"])
        # File receipt is scientific preparation, NOT an approved gate.
        self.assertEqual(analysis.WEIGHT_REPORT_SHA,"")

    def test_archive_contains_no_private_text_or_answers(self):
        d=json.loads(REPORT.read_text())
        forbidden={"article","passage","prompt","gold","foil","reviewer",
                   "question","answer_key","source_text","private_key","human_name"}
        def walk(obj):
            if isinstance(obj,dict):
                self.assertFalse(forbidden.intersection(obj))
                for value in obj.values():walk(value)
            elif isinstance(obj,list):
                for value in obj:walk(value)
        walk(d)

if __name__=="__main__":
    unittest.main()
