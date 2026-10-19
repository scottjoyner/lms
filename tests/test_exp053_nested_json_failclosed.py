"""Untrusted nested JSON must NEVER escape source-free CLI NO-GO handling."""

import base64
import contextlib
import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

RESEARCH = Path(__file__).resolve().parents[1] / "research"
sys.path.insert(0, str(RESEARCH))
import exp053_custody_boundary_audit as custody
import exp053_custody_negative_challenges as challenges
import exp053_custody_receipt_consistency as receipts
import exp053_custody_cross_bundle_replay as cross
import exp053_receipt_checkpoint_comparison as reference
import exp053_observer_publication_chain as publication

VALID_CLAIMS = (RESEARCH / "EXP053-LENOVO-CUSTODY-CONTROL-CLAIMS-2026-10-18.json").read_bytes()
VALID_EMPTY_BUNDLE = (RESEARCH / "EXP053-LENOVO-UNATTESTED-RECEIPTS-2026-10-18.json").read_bytes()
VALID_EMPTY_REFERENCE = (
    RESEARCH / "EXP053-LENOVO-UNTRUSTED-EMPTY-REFERENCE-2026-10-18.json").read_bytes()
REFERENCE_SHA = hashlib.sha256(VALID_EMPTY_REFERENCE).hexdigest()
DEEP_PARSE = b"[" * 11000 + b"0" + b"]" * 11000
DEEP_CANONICAL = b"[" * 2000 + b"0" + b"]" * 2000


class UntrustedNestedJson(unittest.TestCase):
    def test_all_untrusted_json_helpers_reject_extreme_nesting(self):
        for payload in (DEEP_PARSE, DEEP_CANONICAL):
            with self.subTest(depth=len(payload)):
                with self.assertRaises(custody.Hold):
                    custody.parse(payload)
                # Whether the platform's JSON engine rejects nesting at
                # parse time or canonicalization time, rejection is sanitized.
                try:
                    obj = receipts._strict_load(payload)
                except custody.Hold:
                    pass
                else:
                    try:
                        receipts._canonical(obj)
                    except custody.Hold:
                        pass
                with self.assertRaises(custody.Hold):
                    publication._decode_publication(payload)
        # Canonicalizer also needs an independent guard for values from a
        # JSON parser with a higher nesting limit than json.dumps.
        oversized_tree = 0
        for _ in range(12000):
            oversized_tree = [oversized_tree]
        with self.assertRaises(custody.Hold):
            receipts._canonical(oversized_tree)

    def test_all_cli_paths_sanitize_deep_claims_without_debug_trace(self):
        with tempfile.TemporaryDirectory(prefix="exp053-nesting-private-") as tmp:
            root = Path(tmp)
            claims = root / "private-claims.json"
            bundles = root / "private-bundles.json"
            snapshot = root / "private-snapshot.json"
            observed = root / "private-publication.json"
            claims.write_bytes(DEEP_PARSE)
            bundles.write_bytes(VALID_EMPTY_BUNDLE)
            snapshot.write_bytes(VALID_EMPTY_REFERENCE)
            observed.write_bytes(b"{}\n")
            cases = (
                (custody.main, ["--input", str(claims)]),
                (challenges.main, ["--input", str(claims)]),
                (receipts.main, ["--claims", str(claims), "--receipts", str(bundles)]),
                (cross.main, ["--claims", str(claims), "--bundles", str(bundles)]),
                (reference.main, ["--claims", str(claims), "--bundles", str(bundles),
                                  "--reference", str(snapshot),
                                  "--expected-reference-sha256", REFERENCE_SHA]),
            )
            for runner, args in cases:
                with self.subTest(runner=runner.__module__):
                    output = io.StringIO()
                    with contextlib.redirect_stdout(output):
                        rc = runner(args)
                    response = json.loads(output.getvalue())
                    self.assertEqual(rc, 2)
                    self.assertEqual(response["decision"], "NO_GO_RESEARCH_ONLY")
                    self.assertFalse(response["scientific_execution_authorized"])
                    self.assertNotIn(str(root), output.getvalue())
                    self.assertNotIn("private-claims", output.getvalue())
                    self.assertNotIn("Traceback", output.getvalue())

    def test_nested_bundle_reference_and_publication_cli_stay_no_go(self):
        # A canonicalization failure can occur even when json.loads itself
        # succeeds, so cover both tiers and all affected CLI entry points.
        with tempfile.TemporaryDirectory(prefix="exp053-nesting-sensitive-") as tmp:
            root = Path(tmp)
            claims = root / "claims.json"
            bundles = root / "receipts.json"
            reference_path = root / "reference.json"
            publication_path = root / "publication.json"
            claims.write_bytes(VALID_CLAIMS)
            bundles.write_bytes(DEEP_PARSE)
            reference_path.write_bytes(VALID_EMPTY_REFERENCE)
            publication_path.write_bytes(DEEP_CANONICAL)
            cases = (
                (receipts.main, ["--claims", str(claims), "--receipts", str(bundles)]),
                (cross.main, ["--claims", str(claims), "--bundles", str(bundles)]),
                (reference.main, ["--claims", str(claims), "--bundles", str(bundles),
                                  "--reference", str(reference_path),
                                  "--expected-reference-sha256", REFERENCE_SHA]),
            )
            for runner, args in cases:
                with self.subTest(module=runner.__module__):
                    output = io.StringIO()
                    with contextlib.redirect_stdout(output):
                        code = runner(args)
                    self.assertEqual(code, 2)
                    self.assertEqual(json.loads(output.getvalue())["decision"],
                                     "NO_GO_RESEARCH_ONLY")
                    self.assertNotIn(str(root), output.getvalue())
            # Exercise the vulnerable reference canonicalization path itself:
            bundles.write_bytes(VALID_EMPTY_BUNDLE)
            reference_path.write_bytes(DEEP_CANONICAL)
            args = ["--claims", str(claims), "--bundles", str(bundles),
                    "--reference", str(reference_path),
                    "--expected-reference-sha256", hashlib.sha256(DEEP_CANONICAL).hexdigest()]
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = reference.main(args)
            self.assertEqual(code, 2)
            self.assertEqual(json.loads(output.getvalue())["status"],
                             "INVALID_OR_INCONSISTENT_UNTRUSTED_REFERENCE")
            self.assertNotIn(str(root), output.getvalue())
            # The observer verifier accepts an explicitly supplied 32-byte
            # test-only key. It still MUST fail on deeply nested publication.
            try:
                from cryptography.hazmat.primitives import serialization
                from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
            except ImportError:
                return
            public = Ed25519PrivateKey.generate().public_key().public_bytes(
                encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw)
            reference_path.write_bytes(VALID_EMPTY_REFERENCE)
            args = ["--claims", str(claims), "--bundles", str(bundles),
                    "--reference", str(reference_path),
                    "--expected-reference-sha256", REFERENCE_SHA,
                    "--publication", str(publication_path),
                    "--observer-public-key-b64", base64.b64encode(public).decode(),
                    "--expected-observer-key-sha256", hashlib.sha256(public).hexdigest(),
                    "--expected-head-sha256", "0" * 64]
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = publication.main(args)
            self.assertEqual(code, 2)
            self.assertEqual(json.loads(output.getvalue())["status"],
                             "INVALID_OR_UNTRUSTED_PUBLICATION")
            self.assertNotIn(str(root), output.getvalue())


if __name__ == "__main__":
    unittest.main()
