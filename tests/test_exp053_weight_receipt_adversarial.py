"""Adversarial fake metadata only; never an actual EXP053 checkpoint or admission."""
import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]/"research"
sys.path.insert(0,str(ROOT))
import exp053_verified_analysis as v
import exp053_weight_identity as w

def fake_receipt():
    scan=json.loads(v.PREFLIGHT.read_text())
    models={}
    for name,cfg in scan["models"].items():
        if name=="MiniCPM-V-4.7-1B":
            filenames=["model.safetensors"]
            weight_index=None
        else:
            filenames=["test-0001.safetensors","test-0002.safetensors"]
            weight_index="b"*64
        shards=[{"filename":fn,"bytes":8192,"sha256":"a"*64}
                for fn in filenames]
        model={"model":name,
               "checkpoint_revision_or_directory":cfg["checkpoint_directory"],
               "config_sha256":cfg["config_sha256"],
               "tokenizer_sha256":cfg["tokenizer_json_sha256"],
               "weight_index_sha256":weight_index,
               "shards":shards,
               "shard_count":len(shards),
               "total_weight_bytes":sum(s["bytes"] for s in shards)}
        models[name]=model
    return scan,models

def fake_report(models):
    # The synthetic signatures below only test parser math, never human review.
    models=copy.deepcopy(models)
    for x in models.values():
        x["checkpoint_receipt_sha256"]=w.sha(x)
    return {"schema":"exp053-source-only-weight-identity-v1",
            "spec_sha256":v.WEIGHT_SPEC_SHA,
            "tokenizer_receipt_sha256":v.PREFLIGHT_SHA,
            "models":models,"neural_forward_passes":0,
            "human_reviewed_articles":0,
            "checkpoint_receipts_sha256":w.sha(models),
            "claim_limit":"File-byte identity only; no proof of runtime execution or grounding."}

class FakeReceiptValidation(unittest.TestCase):
    def setUp(self):
        self.scan,self.models=fake_receipt()
        self.k2="K2-Horizon-3.7B"
        self.mini="MiniCPM-V-4.7-1B"

    def evaluate(self, mutate=None):
        data=copy.deepcopy(self.models)
        report_extra={}
        if mutate:
            mutate(data,report_extra)
        report=fake_report(data)
        report.update(report_extra)
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"synthetic-metadata-NOT-SCIENCE.json"
            path.write_text(json.dumps(report,sort_keys=True))
            hashed=hashlib.sha256(path.read_bytes()).hexdigest()
            with patch.object(v,"WEIGHT_REPORT",path),patch.object(v,"WEIGHT_REPORT_SHA",hashed):
                return v.admitted_weight_inventory(self.scan)

    def test_actual_admission_pin_remains_empty(self):
        self.assertEqual(v.WEIGHT_REPORT_SHA,"")
        with self.assertRaisesRegex(ValueError,"HOLD"):
            v.admitted_weight_inventory(self.scan)

    def test_synthetic_consistent_metadata_only(self):
        result=self.evaluate()
        self.assertEqual(set(result),set(self.scan["models"]))
        self.assertTrue(all(len(x)==64 for x in result.values()))

    def test_extra_public_report_field_denied(self):
        with self.assertRaisesRegex(ValueError,"HOLD"):
            self.evaluate(lambda m,e:e.update({"unexpected_source_text":"PRIVATE"}))

    def test_extra_model_metadata_denied(self):
        with self.assertRaisesRegex(ValueError,"HOLD"):
            self.evaluate(lambda m,e:m[self.k2].update({"raw_passage":"FORBIDDEN"}))

    def test_duplicate_shard_name_denied(self):
        def tamper(m,e):
            m[self.k2]["shards"][1]["filename"]=m[self.k2]["shards"][0]["filename"]
        with self.assertRaisesRegex(ValueError,"HOLD"):
            self.evaluate(tamper)

    def test_traversal_basename_denied(self):
        with self.assertRaisesRegex(ValueError,"HOLD"):
            self.evaluate(lambda m,e:m[self.k2]["shards"][0].update(
                {"filename":"../bad.safetensors"}))

    def test_uppercase_or_short_digest_denied(self):
        for value in ("A"*64,"12345"):
            with self.assertRaisesRegex(ValueError,"HOLD"):
                self.evaluate(lambda m,e:m[self.k2]["shards"][0].update(
                    {"sha256":value}))

    def test_forged_aggregate_size_denied(self):
        with self.assertRaisesRegex(ValueError,"HOLD"):
            self.evaluate(lambda m,e:m[self.k2].update(
                {"total_weight_bytes":8193}))

    def test_boolean_as_integer_denied(self):
        for value in (True,False,8192.0):
            with self.assertRaisesRegex(ValueError,"HOLD"):
                self.evaluate(lambda m,e:m[self.k2]["shards"][0].update(
                    {"bytes":value}))

    def test_nonsensical_missing_index_denied(self):
        with self.assertRaisesRegex(ValueError,"HOLD"):
            self.evaluate(lambda m,e:m[self.k2].update(
                {"weight_index_sha256":None}))

    def test_mini_spurious_index_denied(self):
        with self.assertRaisesRegex(ValueError,"HOLD"):
            self.evaluate(lambda m,e:m[self.mini].update(
                {"weight_index_sha256":"b"*64}))

    def test_extra_shard_field_denied(self):
        with self.assertRaisesRegex(ValueError,"HOLD"):
            self.evaluate(lambda m,e:m[self.mini]["shards"][0].update(
                {"sample_tokens":[1,2,3]}))

    def test_tokenizer_swap_denied(self):
        with self.assertRaisesRegex(ValueError,"HOLD"):
            self.evaluate(lambda m,e:m[self.k2].update(
                {"tokenizer_sha256":self.models[self.mini]["tokenizer_sha256"]}))

if __name__=="__main__":unittest.main()
