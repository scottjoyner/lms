"""Negative-only historical checkpoint comparisons with ephemeral synthetic keys."""

import base64
import contextlib
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

ROOT = Path(__file__).resolve().parents[1] / "research"
sys.path.insert(0, str(ROOT))
import exp053_custody_boundary_audit as custody
import exp053_custody_negative_challenges as challenges
import exp053_custody_receipt_consistency as receipts
import exp053_receipt_checkpoint_comparison as comparator

try:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
except ImportError:
    serialization = None
    Ed25519PrivateKey = None

CLAIMS = (ROOT / "EXP053-LENOVO-CUSTODY-CONTROL-CLAIMS-2026-10-18.json").read_bytes()
EMPTY_BUNDLE = (ROOT / "EXP053-LENOVO-UNATTESTED-RECEIPTS-2026-10-18.json").read_bytes()
EMPTY_REF = (ROOT / "EXP053-LENOVO-UNTRUSTED-EMPTY-REFERENCE-2026-10-18.json").read_bytes()
NOW = datetime(2026, 10, 18, 20, 0, tzinfo=timezone.utc)


def encode(obj):
    return receipts._canonical(obj)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


class EmptyReference(unittest.TestCase):
    def test_frozen_spec_hash_exact(self):
        self.assertEqual(sha(comparator.SPEC.read_bytes()), comparator.SPEC_SHA256)

    def test_lenovo_empty_reference_makes_no_custody_claim(self):
        report = comparator.compare(CLAIMS, [EMPTY_BUNDLE], EMPTY_REF, sha(EMPTY_REF), now=NOW)
        self.assertEqual(report["prior_receipt_count"], 0)
        self.assertEqual(report["current_untrusted_receipt_count"], 0)
        self.assertEqual(report["additional_untrusted_receipt_count"], 0)
        self.assertTrue(report["historical_receipts_consistent"])
        self.assertEqual(report["reference_custody"], "CALLER_SUPPLIED_UNTRUSTED")
        self.assertEqual(report["durable_replay_protection"], "NOT_PROVEN")
        self.assertEqual(report["decision"], "NO_GO_RESEARCH_ONLY")
        self.assertFalse(report["independent_administration_approved"])
        self.assertFalse(report["witness_durability_approved"])
        self.assertFalse(report["scientific_execution_authorized"])

    def test_wrong_supplied_sha_and_missing_pin_refused(self):
        for pin in (None, "", "0" * 64, "X" * 64, 44, b"a" * 64):
            with self.subTest(pin=pin), self.assertRaises(custody.Hold):
                comparator.compare(CLAIMS, [EMPTY_BUNDLE], EMPTY_REF, pin, now=NOW)

    def test_reference_reencoding_even_with_matching_pin_refused(self):
        obj = json.loads(EMPTY_REF)
        for variant in (encode(obj), json.dumps(obj, indent=4).encode(),
                        encode(obj) + b"\n\n"):
            with self.subTest(n=len(variant)), self.assertRaisesRegex(
                    custody.Hold, "noncanonical"):
                comparator.compare(CLAIMS, [EMPTY_BUNDLE], variant, sha(variant), now=NOW)

    def test_invalid_reference_shapes_and_digests_refused(self):
        obj = json.loads(EMPTY_REF)
        for change in ({"checkpoint_sequence": True}, {"checkpoint_sequence": 1},
                       {"checkpoint_sequence": -1}, {"plan_sha256": "a" * 64},
                       {"candidate_alias": "xwing"}, {"schema": "nope"},
                       {"extra": "private"}, {"receipt_entries": "nope"}):
            data = {**obj, **change}
            raw = encode(data) + b"\n"
            with self.subTest(change=change), self.assertRaises(custody.Hold):
                comparator.compare(CLAIMS, [EMPTY_BUNDLE], raw, sha(raw), now=NOW)
        payload = EMPTY_REF[:-2] + b',"schema":"duplicate"}\n'
        with self.assertRaises(custody.Hold):
            comparator.compare(CLAIMS, [EMPTY_BUNDLE], payload, sha(payload), now=NOW)

    def test_invalid_plan_or_bundle_refused(self):
        with self.assertRaises(custody.Hold):
            comparator.compare(b"{}", [EMPTY_BUNDLE], EMPTY_REF, sha(EMPTY_REF), now=NOW)
        with self.assertRaises(custody.Hold):
            comparator.compare(CLAIMS, [b"{}"], EMPTY_REF, sha(EMPTY_REF), now=NOW)
        with self.assertRaises(custody.Hold):
            comparator.compare(CLAIMS, [EMPTY_BUNDLE]*13, EMPTY_REF, sha(EMPTY_REF), now=NOW)

    def test_frozen_spec_drift_refused(self):
        with patch.object(comparator, "SPEC_SHA256", "a" * 64):
            with self.assertRaisesRegex(custody.Hold, "frozen"):
                comparator.compare(CLAIMS, [EMPTY_BUNDLE], EMPTY_REF, sha(EMPTY_REF), now=NOW)

    def test_cli_no_go_and_no_path_leakage(self):
        with tempfile.TemporaryDirectory(prefix="exp053-reference-") as tmp:
            folder = Path(tmp)
            manifest = folder / "sensitive-claim-filename"
            receipt = folder / "sensitive-bundle-filename"
            reference = folder / "sensitive-checkpoint-filename"
            manifest.write_bytes(CLAIMS)
            receipt.write_bytes(EMPTY_BUNDLE)
            reference.write_bytes(EMPTY_REF)
            args = ["--claims", str(manifest), "--bundles", str(receipt),
                    "--reference", str(reference), "--expected-reference-sha256", sha(EMPTY_REF)]
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = comparator.main(args)
            self.assertEqual(code, 2)
            self.assertEqual(json.loads(output.getvalue())["decision"], "NO_GO_RESEARCH_ONLY")
            self.assertNotIn(str(folder), output.getvalue())
            self.assertNotIn("sensitive", output.getvalue())

    def test_symlink_and_fifo_denied_not_blocked(self):
        with tempfile.TemporaryDirectory(prefix="exp053-reference-") as tmp:
            folder = Path(tmp)
            ref = folder / "normal"
            ref.write_bytes(EMPTY_REF)
            sym = folder / "link"
            sym.symlink_to(ref)
            pipe = folder / "fifo"
            os.mkfifo(pipe)
            manifest = folder / "manifest"
            manifest.write_bytes(CLAIMS)
            receipt = folder / "receipt"
            receipt.write_bytes(EMPTY_BUNDLE)
            for check in (sym, pipe):
                with self.subTest(name=check.name):
                    output = io.StringIO()
                    with contextlib.redirect_stdout(output):
                        code = comparator.main(["--claims", str(manifest),
                            "--bundles", str(receipt), "--reference", str(check),
                            "--expected-reference-sha256", sha(EMPTY_REF)])
                    self.assertEqual(code, 2)
                    report = json.loads(output.getvalue())
                    self.assertEqual(report["status"], "INVALID_OR_INCONSISTENT_UNTRUSTED_REFERENCE")
                    self.assertFalse(report["scientific_execution_authorized"])
                    self.assertFalse(report["witness_durability_approved"])
                    self.assertNotIn(tmp, output.getvalue())


@unittest.skipUnless(Ed25519PrivateKey is not None, "synthetic Ed25519 cryptography optional")
class SignedSyntheticReferences(unittest.TestCase):
    def setUp(self):
        self.private = Ed25519PrivateKey.generate()  # disposable synthetic TEST ONLY
        self.pub = base64.b64encode(self.private.public_key().public_bytes(
            encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw)).decode()
        self.plan = challenges.plan(CLAIMS)
        self.ids = [row["scenario_id"] for row in self.plan["scenario_rows"]]
        self.plan_sha = sha(encode(self.plan))

    def signed_collection(self, scenario_ids, *, outcomes=None, start=0, checkpoint="c"*64):
        data = {"schema": receipts.BUNDLE_SCHEMA,
                "candidate_alias": "lenovo", "plan_sha256": self.plan_sha,
                "observer_public_key_b64": self.pub,
                "checkpoint_reference_sha256": checkpoint, "receipts": []}
        for i, scenario in enumerate(scenario_ids):
            row = {"scenario_id": scenario,
                   "outcome": outcomes[i] if outcomes is not None else "DENIED",
                   "nonce_hex": f"{start + i+1:064x}",
                   "evidence_sha256": "d" * 64,
                   "observed_at_utc": NOW.strftime("%Y-%m-%dT%H:%M:%SZ")}
            unsigned = {"schema": receipts.RECEIPT_SCHEMA,
                        "plan_sha256": self.plan_sha, "candidate_alias": "lenovo",
                        "checkpoint_reference_sha256": checkpoint, **row}
            row["signature_hex"] = self.private.sign(receipts.DOMAIN + encode(unsigned)).hex()
            data["receipts"].append(row)
        return encode(data)

    def checkpoint(self, bundle_bytes, retained_ids):
        rows = {}
        for raw in bundle_bytes:
            for row in json.loads(raw)["receipts"]:
                rows[row["scenario_id"]] = sha(encode(row))
        obj = {"schema": comparator.REFERENCE_SCHEMA,
               "candidate_alias": "lenovo", "plan_sha256": self.plan_sha,
               "checkpoint_sequence": len(retained_ids),
               "receipt_entries": [{"scenario_id": scenario, "receipt_sha256": rows[scenario]}
                                   for scenario in self.ids if scenario in retained_ids]}
        return encode(obj)+b"\n"

    def test_historical_prefix_plus_new_scenarios_consistent_but_not_authoritative(self):
        old = self.signed_collection(self.ids[:3])
        fresh = self.signed_collection(self.ids[3:5], start=3)
        reference = self.checkpoint([old], set(self.ids[:3]))
        got = comparator.compare(CLAIMS, [old, fresh], reference, sha(reference), now=NOW)
        self.assertEqual(got["prior_receipt_count"], 3)
        self.assertEqual(got["current_untrusted_receipt_count"], 5)
        self.assertEqual(got["additional_untrusted_receipt_count"], 2)
        self.assertTrue(got["historical_receipts_consistent"])
        self.assertEqual(got["durable_replay_protection"], "NOT_PROVEN")
        self.assertFalse(got["independent_administration_approved"])
        self.assertFalse(got["scientific_execution_authorized"])
        self.assertNotIn(self.pub, json.dumps(got))
        self.assertNotIn(sha(reference), json.dumps(got))

    def test_omitted_historical_signed_row_detected(self):
        old = self.signed_collection(self.ids[:2])
        reference = self.checkpoint([old], set(self.ids[:2]))
        shortened = self.signed_collection(self.ids[:1])
        with self.assertRaisesRegex(custody.Hold, "omitted"):
            comparator.compare(CLAIMS, [shortened], reference, sha(reference), now=NOW)

    def test_replacement_signed_by_same_untrusted_key_detected(self):
        old = self.signed_collection(self.ids[:1], outcomes=["DENIED"])
        reference = self.checkpoint([old], set(self.ids[:1]))
        other = self.signed_collection(self.ids[:1], outcomes=["BYPASS"])
        # Both bundles verify individually against the SAME synthetic key.
        with self.assertRaisesRegex(custody.Hold, "changed"):
            comparator.compare(CLAIMS, [other], reference, sha(reference), now=NOW)

    def test_forged_signature_and_stale_self_report_refused(self):
        old = self.signed_collection(self.ids[:1])
        reference = self.checkpoint([old], set(self.ids[:1]))
        bogus = json.loads(old)
        bogus["receipts"][0]["signature_hex"] = "0" * 128
        with self.assertRaises(custody.Hold):
            comparator.compare(CLAIMS, [encode(bogus)], reference, sha(reference), now=NOW)
        with self.assertRaises(custody.Hold):
            comparator.compare(CLAIMS, [old], reference, sha(reference),
                               now=NOW + timedelta(hours=25))

    def test_reordered_or_duplicated_reference_entries_refused(self):
        old = self.signed_collection(self.ids[:2])
        for altered in ("reversed", "repeated", "missing_digest", "extra"):
            obj = json.loads(self.checkpoint([old], set(self.ids[:2])))
            if altered == "reversed":
                obj["receipt_entries"].reverse()
            elif altered == "repeated":
                obj["receipt_entries"][1] = obj["receipt_entries"][0]
            elif altered == "missing_digest":
                obj["receipt_entries"][0]["receipt_sha256"] = ""
            else:
                obj["receipt_entries"][0]["extra"] = "secret"
            forged = encode(obj)+b"\n"
            with self.subTest(case=altered), self.assertRaises(custody.Hold):
                comparator.compare(CLAIMS, [old], forged, sha(forged), now=NOW)

    def test_complete_twelve_signed_receipts_never_activate_science(self):
        all12 = self.signed_collection(self.ids)
        reference = self.checkpoint([all12], set(self.ids))
        got = comparator.compare(CLAIMS, [all12], reference, sha(reference), now=NOW)
        self.assertEqual(got["prior_receipt_count"], 12)
        self.assertEqual(got["additional_untrusted_receipt_count"], 0)
        self.assertFalse(got["scientific_execution_authorized"])
        self.assertFalse(got["witness_durability_approved"])
        self.assertEqual(got["reference_custody"], "CALLER_SUPPLIED_UNTRUSTED")

    def test_fake_roll_back_both_reference_and_pin_undetectable_and_still_no_go(self):
        old = self.signed_collection(self.ids[:2])
        shortened = self.signed_collection(self.ids[:1])
        old_reference = self.checkpoint([old], set(self.ids[:2]))
        with self.assertRaises(custody.Hold):
            comparator.compare(CLAIMS, [shortened], old_reference, sha(old_reference), now=NOW)
        # Attacker also replaces the caller-controlled reference and its
        # supplied hash. This tool cannot discover the withheld old reference.
        rewritten = self.checkpoint([shortened], set(self.ids[:1]))
        got = comparator.compare(CLAIMS, [shortened], rewritten, sha(rewritten), now=NOW)
        self.assertTrue(got["historical_receipts_consistent"])
        self.assertEqual(got["reference_custody"], "CALLER_SUPPLIED_UNTRUSTED")
        self.assertEqual(got["durable_replay_protection"], "NOT_PROVEN")
        self.assertFalse(got["scientific_execution_authorized"])


if __name__ == "__main__":
    unittest.main()
