"""Synthetic, offline safety checks for public machine-only aggregate export."""
import copy
import json
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research"))
import exp053_agent_qwen_public as public


class PublicAggregate(unittest.TestCase):
    def setUp(self):
        self.private = {
            "schema": "exp053-qwen-agent-only-summary-v2",
            "complete_frozen_cohort": True,
            "human_adjudicated": False,
            "machine_selected_foil": True,
            "semantic_grounding_proven": False,
            "case_count": 64,
            "condition_count": 384,
            "model_checkpoint_receipt_sha256": "a" * 64,
            "runner_sha256": "b" * 64,
            "UNTRUSTED_PRIVATE_ANSWER_TEXT": "PRIVATE_CASE_LABEL",
        }
        self.metrics = {
            x: {"mean": 0.05, "median": 0.1, "positive": 27, "n": 64}
            for x in ("D", "N", "S")
        }

    def export(self):
        return public.public_payload(self.private, self.metrics, report_sha="c" * 64)

    def test_safe_whitelist_never_leaks_private_fields(self):
        value = self.export()
        raw = json.dumps(value)
        self.assertNotIn("PRIVATE_CASE_LABEL", raw)
        self.assertNotIn("UNTRUSTED_PRIVATE_ANSWER_TEXT", raw)
        self.assertEqual(value["human_reviewers"], 0)
        self.assertFalse(value["scientific_grounding_proven"])
        self.assertFalse(value["human_adjudicated"])
        self.assertFalse(value["independent_semantic_foil"])
        self.assertEqual(value["case_count"], 64)
        self.assertEqual(value["source_conditions"], 384)

    def test_false_human_adjudication_is_rejected(self):
        self.private["human_adjudicated"] = True
        with self.assertRaisesRegex(ValueError, "claims"):
            self.export()

    def test_incomplete_cohort_is_rejected(self):
        self.private["case_count"] = 63
        with self.assertRaisesRegex(ValueError, "cohort"):
            self.export()

    def test_boolean_cohort_count_cannot_masquerade_as_integer(self):
        self.private["case_count"] = True
        with self.assertRaises(ValueError):
            self.export()

    def test_nan_or_infinite_metric_rejected(self):
        for unsafe in (math.nan,math.inf,-math.inf):
            with self.subTest(value=str(unsafe)):
                self.metrics["D"]["mean"] = unsafe
                with self.assertRaisesRegex(ValueError, "aggregate"):
                    self.export()

    def test_extra_undocumented_metric_rejected(self):
        self.metrics["unexpected"] = copy.deepcopy(self.metrics["D"])
        with self.assertRaisesRegex(ValueError, "metric inventory"):
            self.export()

    def test_malformed_model_digest_rejected(self):
        self.private["runner_sha256"] = "not-a-digest"
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            self.export()

    def test_wrong_source_condition_count_rejected(self):
        self.private["condition_count"] = 383
        with self.assertRaisesRegex(ValueError, "cohort"):
            self.export()

    def test_positive_count_above_cohort_refused(self):
        self.metrics["D"]["positive"] = 65
        with self.assertRaisesRegex(ValueError, "aggregate"):
            self.export()

    def test_invalid_full_summary_hash_refused(self):
        with self.assertRaisesRegex(ValueError, "checksum"):
            public.public_payload(self.private, self.metrics, report_sha="broken")


if __name__ == "__main__":
    unittest.main()
