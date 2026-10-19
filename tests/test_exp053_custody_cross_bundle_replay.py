"""Synthetic multi-bundle replay, fork and anti-admission checks. No fleet IO."""

import base64
import contextlib
import copy
import hashlib
import io
import json
import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

RESEARCH = Path(__file__).resolve().parents[1] / "research"
sys.path.insert(0, str(RESEARCH))
import exp053_custody_boundary_audit as custody
import exp053_custody_negative_challenges as challenges
import exp053_custody_receipt_consistency as receipts
import exp053_custody_cross_bundle_replay as cross

try:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
except ImportError:
    serialization = None
    Ed25519PrivateKey = None

CLAIMS = (RESEARCH / "EXP053-LENOVO-CUSTODY-CONTROL-CLAIMS-2026-10-18.json").read_bytes()
EMPTY = (RESEARCH / "EXP053-LENOVO-UNATTESTED-RECEIPTS-2026-10-18.json").read_bytes()
NOW = datetime(2026, 10, 18, 20, 30, tzinfo=timezone.utc)


def encode(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


class SourceOnlyBaseline(unittest.TestCase):
    def test_three_protocols_have_original_frozen_hashes(self):
        for module in (custody, challenges, receipts, cross):
            with self.subTest(module=module.__name__):
                self.assertEqual(hashlib.sha256(module.SPEC.read_bytes()).hexdigest(),
                                 module.SPEC_SHA256)

    def test_empty_real_fixture_has_twelve_missing_and_never_admits(self):
        report = cross.audit(CLAIMS, [EMPTY], now=NOW)
        self.assertEqual(report["bundle_count"], 1)
        self.assertEqual(report["unique_scenario_claim_count"], 0)
        self.assertEqual(len(report["missing_scenario_ids"]), 12)
        self.assertEqual(report["durable_replay_protection"], "NOT_PROVEN")
        self.assertFalse(report["all_challenges_independently_attested"])
        self.assertFalse(report["independent_administration_approved"])
        self.assertFalse(report["witness_durability_approved"])
        self.assertFalse(report["scientific_execution_authorized"])

    def test_normalized_empty_baseline_replay_denied(self):
        reformatted = json.dumps(json.loads(EMPTY), indent=8).encode()
        with self.assertRaisesRegex(custody.Hold, "duplicate normalized"):
            cross.audit(CLAIMS, [EMPTY, reformatted], now=NOW)

    def test_invalid_count_and_frozen_spec_drift_denied(self):
        for data in ([], [EMPTY] * 13, (b"garbage",), None, 4):
            with self.subTest(data_type=type(data).__name__), self.assertRaises(custody.Hold):
                cross.audit(CLAIMS, data, now=NOW)
        with patch.object(cross, "SPEC_SHA256", "a" * 64):
            with self.assertRaisesRegex(custody.Hold, "frozen"):
                cross.audit(CLAIMS, [EMPTY], now=NOW)

    def test_invalid_manifest_cannot_be_laundered_via_collection(self):
        with self.assertRaises(custody.Hold):
            cross.audit(b"{}", [EMPTY], now=NOW)
        with self.assertRaises(custody.Hold):
            cross.audit(CLAIMS, [b"{}"], now=NOW)

    def test_cli_never_grants_approval(self):
        with tempfile.TemporaryDirectory(prefix="exp053-collection-") as tmp:
            folder = Path(tmp)
            claims = folder / "private-named-claims.json"
            bundles = folder / "private-named-bundles.json"
            claims.write_bytes(CLAIMS)
            bundles.write_bytes(EMPTY)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = cross.main(["--claims", str(claims), "--bundles", str(bundles)])
        self.assertEqual(code, 2)
        report = json.loads(out.getvalue())
        self.assertEqual(report["decision"], "NO_GO_RESEARCH_ONLY")
        self.assertFalse(report["scientific_execution_authorized"])
        self.assertNotIn("private-named", out.getvalue())
        self.assertNotIn(tmp, out.getvalue())

    def test_untrusted_file_types_fail_closed_and_no_path_leak(self):
        with tempfile.TemporaryDirectory(prefix="exp053-collection-") as tmp:
            folder = Path(tmp)
            claims = folder / "claims.json"
            claims.write_bytes(CLAIMS)
            baseline = folder / "source.json"
            baseline.write_bytes(EMPTY)
            symlink = folder / "symlink.json"
            symlink.symlink_to(baseline)
            pipe = folder / "fifo.json"
            os.mkfifo(pipe)
            for bad in (symlink, pipe):
                with self.subTest(name=bad.name):
                    out = io.StringIO()
                    with contextlib.redirect_stdout(out):
                        result = cross.main(["--claims", str(claims), "--bundles", str(bad)])
                    self.assertEqual(result, 2)
                    response = json.loads(out.getvalue())
                    self.assertEqual(response["status"], "INVALID_OR_UNVERIFIABLE_INPUT")
                    self.assertFalse(response["scientific_execution_authorized"])
                    self.assertFalse(response["independent_administration_approved"])
                    self.assertFalse(response["witness_durability_approved"])
                    self.assertNotIn(str(folder), out.getvalue())


@unittest.skipUnless(Ed25519PrivateKey is not None, "synthetic cryptography unavailable")
class SyntheticAdversarialBundles(unittest.TestCase):
    def setUp(self):
        self.key = Ed25519PrivateKey.generate()  # synthetic ephemeral key, NOT witness
        self.public = base64.b64encode(self.key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw)).decode()
        self.plans = challenges.plan(CLAIMS)
        self.sha = hashlib.sha256(encode(self.plans)).hexdigest()
        self.ids = [item["scenario_id"] for item in self.plans["scenario_rows"]]
        self.checkpoint = "f" * 64  # no independently published checkpoint

    def fixture(self, positions, *, key=None, public=None, checkpoint=None,
                outcome="DENIED", start=0, timestamp=None):
        signer = self.key if key is None else key
        pub = self.public if public is None else public
        chk = self.checkpoint if checkpoint is None else checkpoint
        data = {
            "schema": receipts.BUNDLE_SCHEMA,
            "plan_sha256": self.sha,
            "candidate_alias": "lenovo",
            "observer_public_key_b64": pub,
            "checkpoint_reference_sha256": chk,
            "receipts": [],
        }
        for count, scenario in enumerate(positions):
            row = {
                "scenario_id": scenario,
                "outcome": outcome,
                "nonce_hex": f"{start + count + 1:064x}",
                "evidence_sha256": "e" * 64,
                "observed_at_utc": (
                    NOW.strftime("%Y-%m-%dT%H:%M:%SZ") if timestamp is None else timestamp),
            }
            unsigned = {
                "schema": receipts.RECEIPT_SCHEMA,
                "plan_sha256": self.sha,
                "candidate_alias": data["candidate_alias"],
                "checkpoint_reference_sha256": chk,
                **row,
            }
            row["signature_hex"] = signer.sign(receipts.DOMAIN + encode(unsigned)).hex()
            data["receipts"].append(row)
        return encode(data)

    def test_disjoint_twelve_synthetic_receipts_cross_bundles_still_no_go(self):
        first = self.fixture(self.ids[:6], start=0)
        second = self.fixture(self.ids[6:], start=6)
        report = cross.audit(CLAIMS, [first, second], now=NOW)
        self.assertEqual(report["unique_scenario_claim_count"], 12)
        self.assertEqual(report["bundle_count"], 2)
        self.assertEqual(report["missing_scenario_ids"], [])
        self.assertEqual(report["claimed_denied_count"], 12)
        self.assertEqual(report["signer_authority"], "UNTRUSTED_SELF_SUPPLIED_KEY")
        self.assertEqual(report["decision"], "NO_GO_RESEARCH_ONLY")
        self.assertFalse(report["scientific_execution_authorized"])
        self.assertFalse(report["all_challenges_independently_attested"])
        self.assertFalse(report["witness_durability_approved"])
        self.assertFalse(report["independent_administration_approved"])
        for secret in (self.public, self.checkpoint,
                       json.loads(first)["receipts"][0]["nonce_hex"],
                       json.loads(first)["receipts"][0]["signature_hex"]):
            self.assertNotIn(secret, json.dumps(report))

    def test_replayed_exact_bundle_or_json_reformat_denied(self):
        bundle = self.fixture(self.ids[:1])
        alternate = json.dumps(json.loads(bundle), indent=4).encode()
        for copied in (bundle, alternate):
            with self.subTest(len=len(copied)), self.assertRaisesRegex(
                    custody.Hold, "duplicate normalized"):
                cross.audit(CLAIMS, [bundle, copied], now=NOW)

    def test_duplicate_scenario_is_rejected_even_with_fresh_nonce(self):
        first = self.fixture(self.ids[:1], start=1)
        second = self.fixture(self.ids[:1], start=100)
        with self.assertRaisesRegex(custody.Hold, "repeated scenario"):
            cross.audit(CLAIMS, [first, second], now=NOW)

    def test_cross_bundle_nonce_collision_denied(self):
        first = self.fixture(self.ids[:1], start=7)
        second = self.fixture(self.ids[1:2], start=7)
        with self.assertRaisesRegex(custody.Hold, "replayed nonce"):
            cross.audit(CLAIMS, [first, second], now=NOW)

    def test_repeated_identical_signed_row_in_new_bundle_denied(self):
        first = self.fixture(self.ids[:1], start=0)
        second = self.fixture(self.ids[1:2], start=100)
        # A modified row is no longer signed, so the frozen parent verifier
        # must already reject this before the cross-bundle check.
        tampered = json.loads(second)
        tampered["receipts"][0]["signature_hex"] = json.loads(first)["receipts"][0]["signature_hex"]
        with self.assertRaises(custody.Hold):
            cross.audit(CLAIMS, [first, encode(tampered)], now=NOW)

    def test_observer_key_rotation_in_same_collection_is_unauthorized(self):
        first = self.fixture(self.ids[:1], start=0)
        newkey = Ed25519PrivateKey.generate()
        newpub = base64.b64encode(newkey.public_key().public_bytes(
            encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw)).decode()
        second = self.fixture(self.ids[1:2], start=9, key=newkey, public=newpub)
        with self.assertRaisesRegex(custody.Hold, "changed observer"):
            cross.audit(CLAIMS, [first, second], now=NOW)

    def test_checkpoint_fork_under_same_key_rejected(self):
        first = self.fixture(self.ids[:1], start=0)
        second = self.fixture(self.ids[1:2], start=9, checkpoint="a" * 64)
        with self.assertRaisesRegex(custody.Hold, "checkpoint context"):
            cross.audit(CLAIMS, [first, second], now=NOW)

    def test_claimed_bypass_and_unknown_not_laundered(self):
        first = self.fixture(self.ids[:1], start=0, outcome="BYPASS")
        second = self.fixture(self.ids[1:2], start=1, outcome="UNKNOWN")
        report = cross.audit(CLAIMS, [first, second], now=NOW)
        self.assertEqual(report["claimed_bypass_count"], 1)
        self.assertEqual(report["claimed_unknown_count"], 1)
        self.assertEqual(report["claimed_denied_count"], 0)
        self.assertFalse(report["scientific_execution_authorized"])

    def test_empty_baseline_mixed_with_claims_denied(self):
        with self.assertRaisesRegex(custody.Hold, "empty baseline"):
            cross.audit(CLAIMS, [EMPTY, self.fixture(self.ids[:1])], now=NOW)
        with self.assertRaisesRegex(custody.Hold, "empty baseline"):
            cross.audit(CLAIMS, [self.fixture(self.ids[:1]), EMPTY], now=NOW)

    def test_stale_and_future_receipt_denied_by_original_verifier(self):
        for offset in (-timedelta(hours=25), timedelta(seconds=301)):
            stamp = (NOW+offset).strftime("%Y-%m-%dT%H:%M:%SZ")
            with self.subTest(stamp=stamp), self.assertRaises(custody.Hold):
                cross.audit(CLAIMS, [self.fixture(self.ids[:1], timestamp=stamp)], now=NOW)

    def test_tampered_signature_and_altered_claims_denied(self):
        raw = json.loads(self.fixture(self.ids[:1]))
        changed = copy.deepcopy(raw)
        changed["receipts"][0]["outcome"] = "BYPASS"
        with self.assertRaises(custody.Hold):
            cross.audit(CLAIMS, [encode(changed)], now=NOW)
        with self.assertRaises(custody.Hold):
            cross.audit(b"{}", [encode(raw)], now=NOW)


if __name__ == "__main__":
    unittest.main()
