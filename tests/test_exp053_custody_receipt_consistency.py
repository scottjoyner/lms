"""Synthetic untrusted signatures: NEVER a real EXP053 human or witness receipt."""

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

try:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
except ImportError:
    Ed25519PrivateKey = None
    serialization = None

CLAIMS = (RESEARCH / "EXP053-LENOVO-CUSTODY-CONTROL-CLAIMS-2026-10-18.json").read_bytes()
EMPTY = (RESEARCH / "EXP053-LENOVO-UNATTESTED-RECEIPTS-2026-10-18.json").read_bytes()
NOW = datetime(2026, 10, 18, 20, 0, tzinfo=timezone.utc)


def encode(payload):
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


class EmptyUntrustedLedger(unittest.TestCase):
    def test_spec_and_dependencies_frozen(self):
        self.assertEqual(hashlib.sha256(receipts.SPEC.read_bytes()).hexdigest(),
                         receipts.SPEC_SHA256)
        self.assertEqual(hashlib.sha256(challenges.SPEC.read_bytes()).hexdigest(),
                         challenges.SPEC_SHA256)
        self.assertEqual(hashlib.sha256(custody.SPEC.read_bytes()).hexdigest(),
                         custody.SPEC_SHA256)

    def test_real_lenovo_has_zero_independent_receipts(self):
        report = receipts.assess(CLAIMS, EMPTY, now=NOW)
        self.assertEqual(report["expected_scenario_count"], 12)
        self.assertEqual(report["cryptographically_consistent_untrusted_receipt_count"], 0)
        self.assertEqual(len(report["missing_scenario_ids"]), 12)
        self.assertEqual(report["signer_authority"], "UNTRUSTED_SELF_SUPPLIED_KEY")
        self.assertFalse(report["all_challenges_independently_attested"])
        self.assertFalse(report["scientific_execution_authorized"])
        self.assertFalse(report["witness_durability_approved"])
        self.assertFalse(report["independent_administration_approved"])

    def test_unattested_bundle_with_fabricated_signer_or_checkpoint_rejected(self):
        for field in ("observer_public_key_b64", "checkpoint_reference_sha256"):
            bundle = json.loads(EMPTY)
            bundle[field] = "a" * 64
            with self.subTest(field=field), self.assertRaises(custody.Hold):
                receipts.assess(CLAIMS, encode(bundle), now=NOW)

    def test_empty_bundle_never_returns_success_from_cli(self):
        with tempfile.TemporaryDirectory(prefix="exp053-receipts-") as tmp:
            path = Path(tmp) / "redacted-receipts.json"
            path.write_bytes(EMPTY)
            claims = Path(tmp) / "redacted-claims.json"
            claims.write_bytes(CLAIMS)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                rc = receipts.main(["--claims", str(claims), "--receipts", str(path)])
        self.assertEqual(rc, 2)
        self.assertEqual(json.loads(out.getvalue())["decision"], "NO_GO_RESEARCH_ONLY")
        self.assertNotIn("redacted-receipts", out.getvalue())
        self.assertNotIn(str(tmp), out.getvalue())

    def test_invalid_bundle_and_duplicate_keys_fail_closed(self):
        bad = (b"", b"\xff", b"{", b'{"schema":"a","schema":"b"}',
               EMPTY + b"x" * receipts.MAX_BYTES, b'{"foo":NaN}')
        for case in bad:
            with self.subTest(size=len(case)), self.assertRaises(custody.Hold):
                receipts.assess(CLAIMS, case, now=NOW)

    def test_cannot_invent_plan_or_candidate_alias(self):
        data = json.loads(EMPTY)
        for change in ({"plan_sha256": "0" * 64}, {"candidate_alias": "destroyer"},
                       {"extra": "privacy leak"}, {"receipts": {}}, {"schema": "other"}):
            with self.subTest(change=change), self.assertRaises(custody.Hold):
                receipts.assess(CLAIMS, encode({**data, **change}), now=NOW)

    def test_frozen_spec_drift_fails_closed(self):
        with patch.object(receipts, "SPEC_SHA256", "a" * 64):
            with self.assertRaises(custody.Hold):
                receipts.assess(CLAIMS, EMPTY, now=NOW)

    def test_malformed_utc_and_untrusted_clock_denied(self):
        for utc in ("2026-1-18T20:00:00Z", "2026-10-18T20:00:00+00:00",
                    "2026-02-31T20:00:00Z", "2026-10-18T20:00:60Z", "garbage"):
            with self.subTest(utc=utc), self.assertRaises(custody.Hold):
                receipts._utc(utc)
        for now in (NOW.replace(tzinfo=None), None):
            if now is not None:
                with self.assertRaises(custody.Hold):
                    receipts.assess(CLAIMS, EMPTY, now=now)

    def test_receipt_input_symlinks_fifos_and_oversize_fail_fast(self):
        with tempfile.TemporaryDirectory(prefix="exp053-receipts-") as tmp:
            directory = Path(tmp)
            regular = directory / "source"
            regular.write_bytes(EMPTY)
            symlink = directory / "link"
            symlink.symlink_to(regular)
            fifo = directory / "pipe"
            os.mkfifo(fifo)
            for item in (symlink, fifo):
                with self.subTest(name=item.name):
                    output = io.StringIO()
                    with contextlib.redirect_stdout(output):
                        rc = receipts.main(["--claims", str(
                            RESEARCH / "EXP053-LENOVO-CUSTODY-CONTROL-CLAIMS-2026-10-18.json"),
                            "--receipts", str(item)])
                    self.assertEqual(rc, 2)
                    self.assertEqual(json.loads(output.getvalue())["status"],
                                     "INVALID_OR_UNVERIFIABLE_INPUT")
                    self.assertFalse(json.loads(output.getvalue())["independent_administration_approved"])
                    self.assertFalse(json.loads(output.getvalue())["witness_durability_approved"])
                    self.assertFalse(json.loads(output.getvalue())["scientific_execution_authorized"])
                    self.assertNotIn(str(directory), output.getvalue())


@unittest.skipUnless(Ed25519PrivateKey is not None, "cryptography optional on hosted CI")
class SyntheticUntrustedSignatures(unittest.TestCase):
    def setUp(self):
        self.private_key = Ed25519PrivateKey.generate()  # DISPOSABLE SOFTWARE FIXTURE
        self.public = self.private_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw)
        self.plan = challenges.plan(CLAIMS)
        self.digest = hashlib.sha256(encode(self.plan)).hexdigest()
        self.signed = {
            "schema": receipts.BUNDLE_SCHEMA,
            "candidate_alias": self.plan["candidate_alias"],
            "plan_sha256": self.digest,
            "checkpoint_reference_sha256": "e" * 64,  # untrusted fake checkpoint
            "observer_public_key_b64": base64.b64encode(self.public).decode(),
            "receipts": [],
        }

    def add(self, scenario, outcome="DENIED", *, nonce=None, time=None):
        row = {
            "scenario_id": scenario,
            "outcome": outcome,
            "nonce_hex": nonce if nonce is not None else (f"{len(self.signed['receipts'])+1:064x}"),
            "evidence_sha256": "f" * 64,  # no actual evidence files
            "observed_at_utc": time if time is not None else NOW.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        unsigned = {
            "schema": receipts.RECEIPT_SCHEMA,
            "plan_sha256": self.digest,
            "candidate_alias": self.signed["candidate_alias"],
            "checkpoint_reference_sha256": self.signed["checkpoint_reference_sha256"],
            **row,
        }
        row["signature_hex"] = self.private_key.sign(
            receipts.DOMAIN + encode(unsigned)).hex()
        self.signed["receipts"].append(row)

    def eval(self, data=None, *, now=NOW):
        return receipts.assess(CLAIMS, encode(self.signed if data is None else data), now=now)

    def test_all_twelve_valid_synthetic_signatures_never_grant_custody(self):
        for item in self.plan["scenario_rows"]:
            self.add(item["scenario_id"])
        report = self.eval()
        self.assertEqual(report["cryptographically_consistent_untrusted_receipt_count"], 12)
        self.assertEqual(report["missing_scenario_ids"], [])
        self.assertEqual(report["claimed_denied_count"], 12)
        self.assertFalse(report["scientific_execution_authorized"])
        self.assertFalse(report["all_challenges_independently_attested"])
        self.assertFalse(report["independent_administration_approved"])
        self.assertFalse(report["witness_durability_approved"])
        self.assertEqual(report["signer_authority"], "UNTRUSTED_SELF_SUPPLIED_KEY")
        self.assertNotIn(self.signed["observer_public_key_b64"], json.dumps(report))
        self.assertNotIn(self.signed["checkpoint_reference_sha256"], json.dumps(report))
        self.assertNotIn(self.signed["receipts"][0]["signature_hex"], json.dumps(report))
        self.assertNotIn(self.signed["receipts"][0]["evidence_sha256"], json.dumps(report))

    def test_partial_receipts_dont_fill_gaps(self):
        self.add(self.plan["scenario_rows"][0]["scenario_id"])
        report = self.eval()
        self.assertEqual(report["cryptographically_consistent_untrusted_receipt_count"], 1)
        self.assertEqual(len(report["missing_scenario_ids"]), 11)

    def test_claimed_bypass_exposes_risk_not_approval(self):
        self.add("SIGNER_READ_OR_USE", outcome="BYPASS")
        report = self.eval()
        self.assertEqual(report["claimed_bypass_count"], 1)
        self.assertFalse(report["scientific_execution_authorized"])

    def test_invalid_signature_and_forged_field_denied(self):
        self.add("SIGNER_READ_OR_USE")
        mutated = copy.deepcopy(self.signed)
        mutated["receipts"][0]["outcome"] = "BYPASS"
        with self.assertRaises(custody.Hold):
            self.eval(mutated)
        mutated = copy.deepcopy(self.signed)
        mutated["receipts"][0]["signature_hex"] = "0" * 128
        with self.assertRaises(custody.Hold):
            self.eval(mutated)
        mutated = copy.deepcopy(self.signed)
        mutated["checkpoint_reference_sha256"] = "a" * 64
        with self.assertRaises(custody.Hold):
            self.eval(mutated)

    def test_swapped_public_key_not_authoritative(self):
        self.add("SIGNER_READ_OR_USE")
        other = Ed25519PrivateKey.generate().public_key().public_bytes(
            encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw)
        swapped = copy.deepcopy(self.signed)
        swapped["observer_public_key_b64"] = base64.b64encode(other).decode()
        with self.assertRaises(custody.Hold):
            self.eval(swapped)

    def test_duplicate_scenarios_and_replayed_nonce_denied(self):
        self.add("SIGNER_READ_OR_USE")
        self.add("FLEET_SIGNER_ACCESS")
        dup = copy.deepcopy(self.signed)
        dup["receipts"].append(copy.deepcopy(dup["receipts"][0]))
        with self.assertRaises(custody.Hold):
            self.eval(dup)
        replay = copy.deepcopy(self.signed)
        replay["receipts"][1]["nonce_hex"] = replay["receipts"][0]["nonce_hex"]
        with self.assertRaises(custody.Hold):
            self.eval(replay)

    def test_stale_future_age_bounds_and_missing_calendar(self):
        for offset, expected in ((-timedelta(hours=24, seconds=1), "stale"),
                                 (timedelta(seconds=301), "future")):
            with self.subTest(offset=offset):
                self.signed["receipts"].clear()
                self.add("SIGNER_READ_OR_USE", time=(NOW + offset).strftime("%Y-%m-%dT%H:%M:%SZ"))
                with self.assertRaisesRegex(custody.Hold, expected):
                    self.eval()
        self.signed["receipts"].clear()
        self.add("SIGNER_READ_OR_USE", time=(NOW-timedelta(hours=24)).strftime("%Y-%m-%dT%H:%M:%SZ"))
        self.assertEqual(self.eval()["cryptographically_consistent_untrusted_receipt_count"], 1)

    def test_plan_rebinding_or_alternative_signature_context_denied(self):
        self.add("SIGNER_READ_OR_USE")
        changed = copy.deepcopy(self.signed)
        changed["plan_sha256"] = "0" * 64
        with self.assertRaises(custody.Hold):
            self.eval(changed)
        changed = copy.deepcopy(self.signed)
        changed["candidate_alias"] = "destroyer"
        with self.assertRaises(custody.Hold):
            self.eval(changed)
        changed = copy.deepcopy(self.signed)
        changed["receipts"][0]["signature_hex"] = self.private_key.sign(
            encode({"wrong": "domain"})).hex()
        with self.assertRaises(custody.Hold):
            self.eval(changed)

    def test_missing_or_bad_key_encoding_denied(self):
        self.add("SIGNER_READ_OR_USE")
        for key in (None, "!", "a" * 100, base64.b64encode(b"abc").decode()):
            with self.subTest(key_len=len(key) if key else 0):
                data = copy.deepcopy(self.signed)
                data["observer_public_key_b64"] = key
                with self.assertRaises(custody.Hold):
                    self.eval(data)


if __name__ == "__main__":
    unittest.main()
