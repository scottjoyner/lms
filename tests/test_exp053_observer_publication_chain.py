"""Synthetic observer publication chains, NEVER independent custody evidence."""

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
import exp053_observer_publication_chain as pub
import exp053_receipt_checkpoint_comparison as comparison

try:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
except ImportError:
    Ed25519PrivateKey = None
    serialization = None

NOW = datetime(2026, 10, 18, 21, 40, tzinfo=timezone.utc)
CLAIMS = (RESEARCH / "EXP053-LENOVO-CUSTODY-CONTROL-CLAIMS-2026-10-18.json").read_bytes()
EMPTY = (RESEARCH / "EXP053-LENOVO-UNATTESTED-RECEIPTS-2026-10-18.json").read_bytes()
EMPTY_REF = (RESEARCH / "EXP053-LENOVO-UNTRUSTED-EMPTY-REFERENCE-2026-10-18.json").read_bytes()


def encode(value):
    return receipts._canonical(value)


def sha(value):
    return hashlib.sha256(value).hexdigest()


class ZeroAuthorityTests(unittest.TestCase):
    def test_frozen_protocol_and_no_production_pins(self):
        self.assertEqual(sha(pub.SPEC.read_bytes()), pub.SPEC_SHA256)
        self.assertEqual(pub.DOMAIN, b"EXP053-OBSERVER-PUBLICATION-V1\x00")
        import exp053_external_witness as actual
        self.assertEqual(actual.APPROVED_EXTERNAL_WITNESS_PUBLIC_KEY_SHA256, "")
        self.assertEqual(actual.APPROVED_EXTERNAL_WITNESS_ADAPTER_SHA256, "")
        self.assertEqual(actual.APPROVED_WITNESS_OPERATOR_PUBLIC_KEY_SHA256, "")

    def test_missing_required_roots_always_reject(self):
        for public, key_sha, head_sha in (("", "a" * 64, "b" * 64),
                                          (None, "a" * 64, "b" * 64),
                                          ("a", "", "b" * 64),
                                          ("a", "a" * 64, ""),
                                          ("a", "a" * 64, "A" * 64)):
            with self.subTest(case=(public, key_sha, head_sha)), self.assertRaises(custody.Hold):
                pub.verify(CLAIMS, [EMPTY], EMPTY_REF, sha(EMPTY_REF), b"{}",
                           public, key_sha, head_sha, now=NOW)

    def test_bad_or_oversized_publication_never_returns_success(self):
        for raw in (b"", b"{}", b"\xff", b'{"a":1,"a":2}',
                    b"x" * (pub.MAX_PUBLICATION_BYTES + 1)):
            with self.subTest(length=len(raw)), self.assertRaises(custody.Hold):
                pub._decode_publication(raw)

    def test_cli_invalid_input_keeps_private_path_out_of_output(self):
        with tempfile.TemporaryDirectory(prefix="exp053-publisher-") as tmp:
            folder = Path(tmp)
            claims = folder / "secret-claims"
            bundles = folder / "secret-bundles"
            reference = folder / "secret-reference"
            publication = folder / "secret-publication"
            claims.write_bytes(CLAIMS)
            bundles.write_bytes(EMPTY)
            reference.write_bytes(EMPTY_REF)
            publication.write_bytes(b"{}")
            cmd = ["--claims", str(claims), "--bundles", str(bundles),
                   "--reference", str(reference), "--publication", str(publication),
                   "--expected-reference-sha256", sha(EMPTY_REF),
                   "--observer-public-key-b64", "bad", "--expected-observer-key-sha256", "a" * 64,
                   "--expected-head-sha256", "b" * 64]
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = pub.main(cmd)
        self.assertEqual(code, 2)
        report = json.loads(output.getvalue())
        self.assertEqual(report["status"], "INVALID_OR_UNTRUSTED_PUBLICATION")
        self.assertFalse(report["independent_administration_approved"])
        self.assertFalse(report["witness_durability_approved"])
        self.assertFalse(report["scientific_execution_authorized"])
        self.assertNotIn("secret", output.getvalue())
        self.assertNotIn(tmp, output.getvalue())


@unittest.skipUnless(Ed25519PrivateKey is not None, "test-only Ed25519 crypto optional")
class SyntheticPublicationCases(unittest.TestCase):
    def setUp(self):
        self.private = Ed25519PrivateKey.generate()  # TEST ONLY; ephemeral
        self.raw_public = self.private.public_key().public_bytes(
            encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw)
        self.b64 = base64.b64encode(self.raw_public).decode()
        self.key_pin = sha(self.raw_public)
        self.plan = challenges.plan(CLAIMS)
        self.plan_sha = sha(encode(self.plan))
        self.scenario = self.plan["scenario_rows"][0]["scenario_id"]

    def signed_observer(self, steps, *, key=None, timestamps=None, references=None):
        signing_key = key if key is not None else self.private
        if references is None:
            references = [sha(EMPTY_REF)] * steps
        if timestamps is None:
            timestamps = [NOW.strftime("%Y-%m-%dT%H:%M:%SZ")] * steps
        obj = {"schema": pub.SCHEMA, "candidate_alias": "lenovo",
               "plan_sha256": self.plan_sha, "checkpoints": []}
        previous = pub.ZERO
        for i in range(steps):
            unsigned = {"schema": pub.SIGNED_SCHEMA, "candidate_alias": "lenovo",
                        "plan_sha256": self.plan_sha,
                        "sequence": i,
                        "previous_statement_sha256": previous,
                        "reference_sha256": references[i],
                        "observed_at_utc": timestamps[i]}
            statement = {k: v for k, v in unsigned.items() if k not in (
                "schema", "candidate_alias", "plan_sha256")}
            statement["signature_hex"] = signing_key.sign(pub.DOMAIN + encode(unsigned)).hex()
            previous = sha(encode({**unsigned, "signature_hex": statement["signature_hex"]}))
            obj["checkpoints"].append(statement)
        return obj, previous

    def verify(self, publication, head, *, claimed_key=None, key_pin=None,
               bundles=None, reference=None, now=NOW, claims=CLAIMS):
        reference = EMPTY_REF if reference is None else reference
        return pub.verify(
            claims, [EMPTY] if bundles is None else bundles,
            reference, sha(reference), encode(publication)+b"\n",
            self.b64 if claimed_key is None else claimed_key,
            self.key_pin if key_pin is None else key_pin, head, now=now)

    def receipt_and_reference(self):
        row = {"scenario_id": self.scenario,
               "outcome": "DENIED", "nonce_hex": "1" * 64,
               "evidence_sha256": "d" * 64,
               "observed_at_utc": NOW.strftime("%Y-%m-%dT%H:%M:%SZ")}
        claim = {"schema": receipts.RECEIPT_SCHEMA, "candidate_alias": "lenovo",
                 "plan_sha256": self.plan_sha, "checkpoint_reference_sha256": "c" * 64,
                 **row}
        row["signature_hex"] = self.private.sign(receipts.DOMAIN + encode(claim)).hex()
        bundle = {"schema": receipts.BUNDLE_SCHEMA, "candidate_alias": "lenovo",
                  "plan_sha256": self.plan_sha, "checkpoint_reference_sha256": "c" * 64,
                  "observer_public_key_b64": self.b64,
                  "receipts": [row]}
        ref = {"schema": comparison.REFERENCE_SCHEMA, "candidate_alias": "lenovo",
               "plan_sha256": self.plan_sha, "checkpoint_sequence": 1,
               "receipt_entries": [{"scenario_id": self.scenario,
                                    "receipt_sha256": sha(encode(row))}]}
        return encode(bundle), encode(ref)+b"\n"

    def test_genesis_empty_reference_is_non_authoritative(self):
        doc, head = self.signed_observer(1)
        got = self.verify(doc, head)
        self.assertEqual(got["observed_head_sequence"], 0)
        self.assertEqual(got["published_statement_count"], 1)
        self.assertTrue(got["head_matches_caller_pin"])
        self.assertTrue(got["observer_public_key_matches_caller_pin"])
        self.assertTrue(got["head_reference_matches_local"])
        self.assertEqual(got["publication_custody"], "CALLER_SUPPLIED_UNTRUSTED")
        self.assertEqual(got["durable_replay_protection"], "NOT_PROVEN")
        self.assertFalse(got["independent_administration_approved"])
        self.assertFalse(got["witness_durability_approved"])
        self.assertFalse(got["scientific_execution_authorized"])
        self.assertEqual(got["decision"], "NO_GO_RESEARCH_ONLY")
        self.assertNotIn(self.b64, json.dumps(got))
        self.assertNotIn(head, json.dumps(got))

    def test_append_signed_checkpoint_to_first_receipt(self):
        bundle, ref = self.receipt_and_reference()
        doc, head = self.signed_observer(2, references=[sha(EMPTY_REF), sha(ref)])
        got = self.verify(doc, head, bundles=[bundle], reference=ref)
        self.assertEqual(got["observed_head_sequence"], 1)
        self.assertEqual(got["published_statement_count"], 2)
        self.assertFalse(got["scientific_execution_authorized"])

    def test_rolled_back_reference_rejected_by_retained_head(self):
        bundle, ref = self.receipt_and_reference()
        current, head = self.signed_observer(2, references=[sha(EMPTY_REF), sha(ref)])
        old, old_head = self.signed_observer(1)
        self.assertNotEqual(head, old_head)
        with self.assertRaisesRegex(custody.Hold, "head"):
            self.verify(old, head)
        self.assertEqual(self.verify(old, old_head)["durable_replay_protection"],
                         "NOT_PROVEN")
        # A correctly retained old head is not an independent trust root.
        self.assertFalse(self.verify(current, head, bundles=[bundle], reference=ref)[
            "scientific_execution_authorized"])

    def test_replacing_entire_chain_key_and_both_pins_is_not_detected(self):
        other = Ed25519PrivateKey.generate()
        other_raw = other.public_key().public_bytes(
            encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw)
        forged_doc, forged_head = self.signed_observer(1, key=other)
        result = self.verify(forged_doc, forged_head,
                             claimed_key=base64.b64encode(other_raw).decode(),
                             key_pin=sha(other_raw))
        self.assertTrue(result["head_matches_caller_pin"])
        self.assertEqual(result["durable_replay_protection"], "NOT_PROVEN")
        self.assertFalse(result["independent_administration_approved"])
        self.assertFalse(result["scientific_execution_authorized"])

    def test_wrong_key_pin_wrong_head_or_swapped_public_key_denied(self):
        doc, head = self.signed_observer(1)
        with self.assertRaises(custody.Hold):
            self.verify(doc, "a" * 64)
        with self.assertRaises(custody.Hold):
            self.verify(doc, head, key_pin="a" * 64)
        with self.assertRaises(custody.Hold):
            self.verify(doc, head, claimed_key=base64.b64encode(b"wrong").decode())

    def test_invalid_sequence_genesis_chain_or_signature_denied(self):
        base, head = self.signed_observer(1)
        for field, changed in (("sequence", True), ("sequence", 1),
                               ("sequence", -1), ("previous_statement_sha256", "a" * 64),
                               ("signature_hex", "0" * 128), ("reference_sha256", "b" * 64)):
            doc = copy.deepcopy(base)
            doc["checkpoints"][0][field] = changed
            with self.subTest(field=field), self.assertRaises(custody.Hold):
                self.verify(doc, head)
        doc, head = self.signed_observer(2)
        doc["checkpoints"][1]["previous_statement_sha256"] = "e" * 64
        with self.assertRaises(custody.Hold):
            self.verify(doc, head)

    def test_malformed_schema_wrong_alias_plan_extra_field_and_whitespace(self):
        doc, head = self.signed_observer(1)
        for field, value in (("schema", "wrong"), ("candidate_alias", "xwing"),
                             ("plan_sha256", "a" * 64), ("checkpoints", []),
                             ("unexpected", "secret")):
            changed = copy.deepcopy(doc)
            changed[field] = value
            with self.subTest(field=field), self.assertRaises(custody.Hold):
                self.verify(changed, head)
        with self.assertRaisesRegex(custody.Hold, "noncanonical"):
            pub._decode_publication(json.dumps(doc, indent=2).encode()+b"\n")

    def test_older_nonmonotonic_future_and_invalid_utc_denied(self):
        old = (NOW - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
        before = (NOW - timedelta(hours=3)).strftime("%Y-%m-%dT%H:%M:%SZ")
        doc, head = self.signed_observer(2, timestamps=[old, before])
        with self.assertRaisesRegex(custody.Hold, "chronology"):
            self.verify(doc, head)
        tomorrow = (NOW + timedelta(seconds=301)).strftime("%Y-%m-%dT%H:%M:%SZ")
        doc, head = self.signed_observer(1, timestamps=[tomorrow])
        with self.assertRaisesRegex(custody.Hold, "chronology"):
            self.verify(doc, head)
        doc, head = self.signed_observer(1)
        doc["checkpoints"][0]["observed_at_utc"] = "2026-02-31T00:00:00Z"
        with self.assertRaises(custody.Hold):
            self.verify(doc, head)

    def test_wrong_reference_sequence_or_changed_reference_denied(self):
        doc, head = self.signed_observer(1)
        bundle, ref = self.receipt_and_reference()
        with self.assertRaisesRegex(custody.Hold, "head"):
            self.verify(doc, head, bundles=[bundle], reference=ref)
        with self.assertRaises(custody.Hold):
            self.verify(doc, head, reference=EMPTY_REF.replace(b'lenovo', b'xwing'))

    def test_cli_valid_synthetic_statement_is_still_exit_two(self):
        publication, head = self.signed_observer(1)
        with tempfile.TemporaryDirectory(prefix="exp053-offline-publisher-") as tmp:
            base = Path(tmp)
            for name, payload in (("claims", CLAIMS), ("bundle", EMPTY),
                                  ("reference", EMPTY_REF),
                                  ("publication", encode(publication) + b"\n")):
                (base / name).write_bytes(payload)
            params = [
                "--claims", str(base / "claims"), "--bundles", str(base / "bundle"),
                "--reference", str(base / "reference"),
                "--expected-reference-sha256", sha(EMPTY_REF),
                "--publication", str(base / "publication"),
                "--observer-public-key-b64", self.b64,
                "--expected-observer-key-sha256", self.key_pin,
                "--expected-head-sha256", head,
            ]
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                result = pub.main(params)
        self.assertEqual(result, 2)
        report = json.loads(out.getvalue())
        self.assertEqual(report["published_statement_count"], 1)
        self.assertEqual(report["publication_custody"], "CALLER_SUPPLIED_UNTRUSTED")
        self.assertFalse(report["scientific_execution_authorized"])
        self.assertNotIn(tmp, out.getvalue())
        self.assertNotIn(self.b64, out.getvalue())

    def test_symlink_and_fifo_publication_sources_fail_without_blocking(self):
        publication, head = self.signed_observer(1)
        with tempfile.TemporaryDirectory(prefix="exp053-publication-unsafe-") as tmp:
            base = Path(tmp)
            for name, payload in (("claims", CLAIMS), ("bundle", EMPTY),
                                  ("reference", EMPTY_REF),
                                  ("publication", encode(publication) + b"\n")):
                (base / name).write_bytes(payload)
            alias = base / "symlink"
            alias.symlink_to(base / "publication")
            fifo = base / "fifo"
            os.mkfifo(fifo)
            for candidate in (alias, fifo):
                with self.subTest(name=candidate.name):
                    params = [
                        "--claims", str(base / "claims"), "--bundles", str(base / "bundle"),
                        "--reference", str(base / "reference"),
                        "--expected-reference-sha256", sha(EMPTY_REF),
                        "--publication", str(candidate),
                        "--observer-public-key-b64", self.b64,
                        "--expected-observer-key-sha256", self.key_pin,
                        "--expected-head-sha256", head,
                    ]
                    out = io.StringIO()
                    with contextlib.redirect_stdout(out):
                        result = pub.main(params)
                    self.assertEqual(result, 2)
                    report = json.loads(out.getvalue())
                    self.assertEqual(report["status"], "INVALID_OR_UNTRUSTED_PUBLICATION")
                    self.assertFalse(report["scientific_execution_authorized"])
                    self.assertNotIn(str(base), out.getvalue())

    def test_duplicate_json_fields_refused_as_ambiguous(self):
        with self.assertRaises(custody.Hold):
            pub._decode_publication(b'{"schema":"x","schema":"y"}\n')

    def test_protocol_drift_denied(self):
        doc, head = self.signed_observer(1)
        with patch.object(pub, "SPEC_SHA256", "0" * 64):
            with self.assertRaisesRegex(custody.Hold, "frozen"):
                self.verify(doc, head)


if __name__ == "__main__":
    unittest.main()
