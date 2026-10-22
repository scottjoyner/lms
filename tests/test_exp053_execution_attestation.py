"""Synthetic cryptographic verification tests, NEVER real scorer evidence."""
import base64
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
import exp053_execution_attestation as witness
from exp053_race_review import canonical

try:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
except ImportError:
    Ed25519PrivateKey = None
    serialization = None

def sha(b):
    return hashlib.sha256(b).hexdigest()

class UnconfiguredRunnerAuthority(unittest.TestCase):
    def test_real_independent_operator_pins_are_empty(self):
        self.assertEqual(witness.APPROVED_RUNNER_PUBLIC_KEY_SHA256, "")
        self.assertEqual(witness.APPROVED_RUNNER_SOURCE_SHA256, "")
        self.assertEqual(witness.filehash(witness.SPEC), witness.SPEC_SHA256)


@unittest.skipUnless(Ed25519PrivateKey is not None, "cryptography optional on hosted CI")
class ScorerSignatureEvidence(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.private = Path(temp.name)
        self.key = Ed25519PrivateKey.generate()  # SOFTWARE FIXTURE ONLY
        self.public = self.key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw)
        self.pub_fingerprint = sha(self.public)
        self.source_fingerprint = "a" * 64  # NO runner implementation exists
        self.packet = "b" * 64
        self.admission = "c" * 64
        self.weight = "d" * 64
        self.tokenizer = "e" * 64
        self.prompt = "f" * 64
        self.checkpoints = {"K2-Horizon-3.7B": "1" * 64,
                            "Qwen3.5-4B-Base": "2" * 64,
                            "MiniCPM-V-4.7-1B": "3" * 64}
        self.score = self.private / "synthetic-scores.json"
        self.log = self.private / "synthetic-attempt-log.json"
        self.sig = self.private / "synthetic-attestation.json"
        self.save(self.score, b"[]\n")
        self.save(self.log, b'{"attempts":[],"status":"synthetic-only"}\n')
        self.record = {
            "schema": witness.SCHEMA,
            "execution_id": "synthetic_not_a_real_run_123",
            "runner_public_key_b64": base64.b64encode(self.public).decode(),
            "runner_source_sha256": self.source_fingerprint,
            "packet_sha256": self.packet,
            "signed_admission_sha256": self.admission,
            "weight_inventory_sha256": self.weight,
            "tokenizer_preflight_sha256": self.tokenizer,
            "prompt_builder_sha256": self.prompt,
            "score_archive_sha256": sha(self.score.read_bytes()),
            "attempt_log_sha256": sha(self.log.read_bytes()),
            "checkpoint_receipts": self.checkpoints,
            "score_row_count": 720,
            "execution_environment_sha256": "4" * 64,
            "runner_mode": "raw", "simulation": False,
        }

    def save(self, path, data):
        path.write_bytes(data)
        os.chmod(path, 0o600)

    def sign(self, data=None):
        source = self.record if data is None else data
        unsigned = {k: v for k, v in source.items() if k != "signature_hex"}
        signed = dict(unsigned)
        signed["signature_hex"] = self.key.sign(canonical(unsigned)).hex()
        self.save(self.sig, json.dumps(signed, sort_keys=True).encode() + b"\n")

    def verify(self, *, pin_root=True):
        with patch.object(witness, "APPROVED_RUNNER_PUBLIC_KEY_SHA256",
                          self.pub_fingerprint if pin_root else ""), patch.object(
            witness, "APPROVED_RUNNER_SOURCE_SHA256", self.source_fingerprint):
            return witness.verify_attestation(
                self.sig, self.score, self.log, self.private,
                packet_sha256=self.packet, admission_sha256=self.admission,
                weight_inventory_sha256=self.weight,
                tokenizer_preflight_sha256=self.tokenizer,
                prompt_builder_sha256=self.prompt,
                checkpoint_receipts=self.checkpoints, approved_count=40)

    def test_real_operator_pins_remain_empty(self):
        self.assertEqual(witness.APPROVED_RUNNER_PUBLIC_KEY_SHA256, "")
        self.assertEqual(witness.APPROVED_RUNNER_SOURCE_SHA256, "")
        self.sign()
        with self.assertRaisesRegex(ValueError, "HOLD"):
            self.verify(pin_root=False)

    def test_valid_synthetic_signature_is_not_human_approval(self):
        self.sign()
        item = self.verify()
        self.assertEqual(item["score_archive_sha256"], sha(self.score.read_bytes()))
        self.assertEqual(item["attempt_log_sha256"], sha(self.log.read_bytes()))
        self.assertEqual(item["score_rows"], 720)
        self.assertNotIn("runner_public_key_b64", item)
        self.assertNotIn("score_text", item)

    def test_same_descriptor_score_bytes_are_parsed(self):
        # A valid digest binds the actual bytes parsed, not a separate file read.
        self.save(self.score, b'[{"case_id":"synthetic_case"}]\n')
        expected = sha(self.score.read_bytes())
        rows = witness.read_attested_scores(self.score, self.private, expected)
        self.assertEqual(rows, [{"case_id": "synthetic_case"}])

    def test_swapped_score_bytes_rejected_by_same_descriptor(self):
        original = sha(self.score.read_bytes())
        self.save(self.score, b'["substituted source-free scores"]\n')
        with self.assertRaisesRegex(ValueError, "HOLD"):
            witness.read_attested_scores(self.score, self.private, original)

    def test_duplicate_json_key_and_nan_denied_during_score_parse(self):
        for bad in (b'[{"case_id":"x","case_id":"y"}]\n',
                    b'[{"logprob":NaN}]\n',
                    b'[{"logprob":Infinity}]\n'):
            self.save(self.score, bad)
            with self.assertRaisesRegex(ValueError, "HOLD"):
                witness.read_attested_scores(self.score, self.private,
                                             sha(self.score.read_bytes()))

    def test_non_list_archive_and_bad_hash_denied(self):
        with self.assertRaisesRegex(ValueError, "HOLD"):
            witness.read_attested_scores(self.score, self.private, "invalid")
        self.save(self.score, b'{"scores":[]}')
        with self.assertRaisesRegex(ValueError, "HOLD"):
            witness.read_attested_scores(self.score, self.private,
                                         sha(self.score.read_bytes()))

    def test_bounded_score_archive_size(self):
        self.save(self.score, b'["synthetic and over configured bound"]')
        with patch.object(witness, "MAX_SCORE_ARCHIVE_BYTES", 24):
            with self.assertRaisesRegex(ValueError, "HOLD"):
                witness.read_attested_scores(self.score, self.private,
                                             sha(self.score.read_bytes()))

    def test_modified_score_bytes_fail(self):
        self.sign()
        self.save(self.score, b'["fabricated output"]\n')
        with self.assertRaisesRegex(ValueError, "HOLD"):
            self.verify()

    def test_modified_attempt_log_bytes_fail(self):
        self.sign()
        self.save(self.log, b'{"attempts":["changed after signature"]}\n')
        with self.assertRaisesRegex(ValueError, "HOLD"):
            self.verify()

    def test_wrong_cohort_or_admission_denied(self):
        for field in ("packet_sha256", "signed_admission_sha256",
                      "weight_inventory_sha256", "tokenizer_preflight_sha256",
                      "prompt_builder_sha256", "runner_source_sha256"):
            self.record[field] = "0" * 64
            self.sign()
            with self.assertRaisesRegex(ValueError, "HOLD"):
                self.verify()
            self.record[field] = self.context_value(field)

    def context_value(self, field):
        return {"packet_sha256": self.packet,
                "signed_admission_sha256": self.admission,
                "weight_inventory_sha256": self.weight,
                "tokenizer_preflight_sha256": self.tokenizer,
                "prompt_builder_sha256": self.prompt,
                "runner_source_sha256": self.source_fingerprint}[field]

    def test_cross_model_checkpoint_substitution_denied(self):
        self.record["checkpoint_receipts"] = dict(self.checkpoints)
        self.record["checkpoint_receipts"]["MiniCPM-V-4.7-1B"] = "f" * 64
        self.sign()
        with self.assertRaisesRegex(ValueError, "HOLD"):
            self.verify()

    def test_wrong_runner_root_denied(self):
        self.sign()
        with patch.object(witness, "APPROVED_RUNNER_PUBLIC_KEY_SHA256", "0" * 64):
            with patch.object(witness, "APPROVED_RUNNER_SOURCE_SHA256", self.source_fingerprint):
                with self.assertRaisesRegex(ValueError, "HOLD"):
                    witness.verify_attestation(
                        self.sig, self.score, self.log, self.private,
                        packet_sha256=self.packet, admission_sha256=self.admission,
                        weight_inventory_sha256=self.weight,
                        tokenizer_preflight_sha256=self.tokenizer,
                        prompt_builder_sha256=self.prompt,
                        checkpoint_receipts=self.checkpoints, approved_count=40)

    def test_tampered_signature_denied(self):
        self.sign()
        d = json.loads(self.sig.read_bytes())
        d["signature_hex"] = "00" * 64
        self.save(self.sig, json.dumps(d).encode())
        with self.assertRaisesRegex(ValueError, "HOLD"):
            self.verify()

    def test_simulation_or_chat_mode_denied_even_signed(self):
        for field, value in (("simulation", True), ("runner_mode", "chat"),
                             ("score_row_count", 718)):
            original = self.record[field]
            self.record[field] = value
            self.sign()
            with self.assertRaisesRegex(ValueError, "HOLD"):
                self.verify()
            self.record[field] = original

    def test_extra_field_denied_even_signed(self):
        self.record["private_passage"] = "source only synthetic"
        self.sign()
        with self.assertRaisesRegex(ValueError, "HOLD"):
            self.verify()

    def test_duplicate_json_key_denied_even_with_valid_root(self):
        self.sign()
        raw = self.sig.read_text().rstrip()
        self.save(self.sig, (raw[:-1] + ',"schema":"' + witness.SCHEMA + '"}').encode())
        with self.assertRaisesRegex(ValueError, "HOLD"):
            self.verify()

    def test_private_file_permissions_and_symlinks_denied(self):
        self.sign()
        os.chmod(self.score, 0o644)
        with self.assertRaisesRegex(ValueError, "HOLD"):
            self.verify()
        os.chmod(self.score, 0o600)
        moved = self.private / "real-score.json"
        moved.write_bytes(self.score.read_bytes())
        os.chmod(moved, 0o600)
        self.score.unlink()
        self.score.symlink_to(moved)
        with self.assertRaisesRegex(ValueError, "HOLD"):
            self.verify()

    def test_missing_attempt_log_denied(self):
        self.sign()
        self.log.unlink()
        with self.assertRaisesRegex(ValueError, "HOLD"):
            self.verify()

    def test_weak_custody_counts_denied(self):
        self.sign()
        with patch.object(witness, "APPROVED_RUNNER_PUBLIC_KEY_SHA256",
                          self.pub_fingerprint), patch.object(
            witness, "APPROVED_RUNNER_SOURCE_SHA256", self.source_fingerprint):
            with self.assertRaisesRegex(ValueError, "HOLD"):
                witness.verify_attestation(
                    self.sig, self.score, self.log, self.private,
                    packet_sha256=self.packet, admission_sha256=self.admission,
                    weight_inventory_sha256=self.weight,
                    tokenizer_preflight_sha256=self.tokenizer,
                    prompt_builder_sha256=self.prompt,
                    checkpoint_receipts=self.checkpoints, approved_count=39)

if __name__ == "__main__":
    unittest.main()
