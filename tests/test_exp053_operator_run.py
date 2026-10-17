"""Synthetic-only independent operator run-claim and replay tests.

Generated Ed25519 fixture keys are NOT owner-approved science authorities.
All persistent writes here occur only under disposable local test directories.
"""
import base64
import hashlib
import json
import os
import sys
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

R = Path(__file__).resolve().parents[1] / "research"
sys.path.insert(0, str(R))
import exp053_operator_run as run
import exp053_external_witness as witness
from exp053_race_review import canonical

try:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
except ImportError:
    Ed25519PrivateKey = None
    serialization = None


def digest(data):
    return hashlib.sha256(data).hexdigest()


@unittest.skipUnless(Ed25519PrivateKey is not None, "crypto optional in lean CI")
class SyntheticOperatorClaim(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.scorer_private = root / "scorer"
        self.operator_dir = root / "operator"
        self.scorer_private.mkdir(mode=0o700)
        self.operator_dir.mkdir(mode=0o700)
        self.state_path = self.operator_dir / "operator-highwater.json"
        self.create_state()
        self.key = Ed25519PrivateKey.generate()
        self.pub = self.key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw)
        self.pubsha = digest(self.pub)
        self.witness_key = Ed25519PrivateKey.generate()
        self.witness_pub = self.witness_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw)
        self.witness_pub_sha = digest(self.witness_pub)
        self.highwater = (0, "0" * 64)
        self.highwater_lock = threading.Lock()
        self.witness_fault = None
        self.packet = "d" * 64
        self.provenance = self.make_provenance("fictional-run-001")
        self.auth_path = self.scorer_private / "operator-auth-001.json"
        self.last_record = None

    def make_provenance(self, name):
        return {"execution_id_sha256": digest(name.encode()),
                "execution_attestation_sha256": "a" * 64,
                "score_archive_sha256": "b" * 64,
                "runner_public_key_sha256": "c" * 64}

    def create_state(self, sequence=0, prior=None):
        state = {"schema": run.STATE_SCHEMA, "last_sequence": sequence,
                 "last_claim_sha256": prior if prior is not None else "0" * 64}
        self.state_path.write_text(json.dumps(state) + "\n")
        os.chmod(self.state_path, 0o600)

    def sign(self, sequence=1, prior=None, provenance=None, path=None,
             **changes):
        prov = self.provenance if provenance is None else provenance
        record = {
            "schema": run.SCHEMA,
            "runner_execution_id_sha256": prov["execution_id_sha256"],
            "execution_attestation_sha256": prov["execution_attestation_sha256"],
            "packet_sha256": self.packet,
            "score_archive_sha256": prov["score_archive_sha256"],
            "operator_custody_reference_sha256": "e" * 64,
            "sequence": sequence,
            "previous_claim_sha256": prior if prior is not None else "0" * 64,
            "operator_public_key_b64": base64.b64encode(self.pub).decode(),
        }
        record.update(changes)
        record["signature_hex"] = self.key.sign(canonical(record)).hex()
        target = self.auth_path if path is None else path
        target.write_text(json.dumps(record, sort_keys=True) + "\n")
        os.chmod(target, 0o600)
        self.last_record = record
        return target

    def synthetic_external_cas(self, request):
        """Isolated fake remote state: NOT independently operated or durable."""
        if self.witness_fault == "offline":
            raise ConnectionError("synthetic witness unavailable")
        from exp053_witness_request_auth import verify_authenticated_request
        verify_authenticated_request(request, self.pubsha)
        with self.highwater_lock:
            count, last = self.highwater
            if (request["schema"] != witness.REQUEST_SCHEMA or
                    request["expected_previous_sequence"] != count or
                    request["previous_claim_sha256"] != last or
                    request["sequence"] != count + 1):
                raise ValueError("mock external atomic CAS conflict")
            self.highwater = (request["sequence"], request["claim_sha256"])
            receipt = {
                "schema": witness.RESPONSE_SCHEMA,
                "nonce_hex": request["nonce_hex"],
                "sequence": request["sequence"],
                "previous_claim_sha256": request["previous_claim_sha256"],
                "claim_sha256": request["claim_sha256"],
                "runner_execution_id_sha256": request["runner_execution_id_sha256"],
                "witness_counter": request["sequence"],
                "status": "committed",
                "witness_public_key_b64": base64.b64encode(self.witness_pub).decode(),
            }
            if self.witness_fault == "stale_nonce":
                receipt["nonce_hex"] = "a" * 64
            elif self.witness_fault == "wrong_counter":
                receipt["witness_counter"] = request["sequence"] + 1
            elif self.witness_fault == "boolean_counter":
                receipt["witness_counter"] = True
            elif self.witness_fault == "not_committed":
                receipt["status"] = "rejected"
            elif self.witness_fault == "wrong_execution":
                receipt["runner_execution_id_sha256"] = "f" * 64
            elif self.witness_fault == "extra_field":
                receipt["private_research"] = "must not be accepted"
            receipt["signature_hex"] = self.witness_key.sign(canonical(receipt)).hex()
            if self.witness_fault == "tampered_signature":
                receipt["signature_hex"] = "00" * 64
            if self.witness_fault == "wrong_public_key":
                receipt["witness_public_key_b64"] = base64.b64encode(
                    Ed25519PrivateKey.generate().public_key().public_bytes(
                        encoding=serialization.Encoding.Raw,
                        format=serialization.PublicFormat.Raw)).decode()
            return receipt

    def consume(self, auth=None, provenance=None, *, pin=True):
        with patch.object(run, "APPROVED_OPERATOR_PUBLIC_KEY_SHA256",
                          self.pubsha if pin else ""), patch.object(
            run, "APPROVED_OPERATOR_CLAIM_LEDGER_PATH", str(self.state_path)), patch.object(
            witness, "APPROVED_EXTERNAL_WITNESS_PUBLIC_KEY_SHA256",
            self.witness_pub_sha), patch.object(
            witness, "APPROVED_EXTERNAL_WITNESS_ADAPTER_SHA256", "e" * 64), patch.object(
            witness, "APPROVED_WITNESS_OPERATOR_PUBLIC_KEY_SHA256", self.pubsha), patch.object(
            witness, "_external_cas", self.synthetic_external_cas):
            return run.consume_unique_run(
                self.auth_path if auth is None else auth,
                self.scorer_private,
                provenance=self.provenance if provenance is None else provenance,
                packet_sha256=self.packet)

    def test_no_real_operator_pins_exist(self):
        self.assertEqual(run.APPROVED_OPERATOR_PUBLIC_KEY_SHA256, "")
        self.assertEqual(run.APPROVED_OPERATOR_CLAIM_LEDGER_PATH, "")
        self.assertEqual(witness.APPROVED_EXTERNAL_WITNESS_PUBLIC_KEY_SHA256, "")
        self.assertEqual(witness.APPROVED_EXTERNAL_WITNESS_ADAPTER_SHA256, "")
        self.assertEqual(witness.APPROVED_WITNESS_OPERATOR_PUBLIC_KEY_SHA256, "")
        self.assertEqual(witness.filehash(witness.SPEC), witness.SPEC_SHA256)
        self.assertEqual(run.filehash(run.SPEC), run.SPEC_SHA256)
        self.assertEqual(run.filehash(run.MARKER_SPEC), run.MARKER_SPEC_SHA256)
        self.sign()
        with self.assertRaisesRegex(ValueError, "HOLD"):
            self.consume(pin=False)

    def test_first_synthetic_run_claim_consumed_exactly_once(self):
        self.sign()
        first = self.consume()
        self.assertEqual(first["operator_sequence"], 1)
        self.assertEqual(first["operator_authorization_sha256"],
                         digest(self.auth_path.read_bytes()))
        self.assertEqual(json.loads(self.state_path.read_text())["last_sequence"], 1)
        marker = self.state_path.with_name(
            self.state_path.name + ".used-" + self.provenance["execution_id_sha256"])
        self.assertTrue(marker.exists())
        self.assertEqual(marker.read_text().strip(),
                         first["operator_authorization_sha256"])
        with self.assertRaisesRegex(ValueError, "HOLD"):
            self.consume()

    def test_new_sequence_cannot_reuse_original_execution_id(self):
        self.sign()
        first = self.consume()
        path2 = self.scorer_private / "operator-auth-002.json"
        self.sign(sequence=2, prior=first["operator_authorization_sha256"], path=path2)
        with self.assertRaisesRegex(ValueError, "HOLD"):
            self.consume(path2)
        self.assertEqual(json.loads(self.state_path.read_text())["last_sequence"], 1)

    def test_new_run_id_can_follow_correct_chain(self):
        self.sign()
        first = self.consume()
        new_prov = self.make_provenance("fictional-run-002")
        path2 = self.scorer_private / "operator-auth-002.json"
        self.sign(sequence=2, prior=first["operator_authorization_sha256"],
                  path=path2, provenance=new_prov)
        second = self.consume(path2, new_prov)
        self.assertEqual(second["operator_sequence"], 2)
        self.assertEqual(json.loads(self.state_path.read_text())["last_sequence"], 2)

    def test_out_of_order_sequence_denied(self):
        self.sign(sequence=2)
        with self.assertRaisesRegex(ValueError, "HOLD"):
            self.consume()

    def test_wrong_previous_claim_denied(self):
        self.sign(prior="f" * 64)
        with self.assertRaisesRegex(ValueError, "HOLD"):
            self.consume()

    def test_wrong_signed_input_denied(self):
        for bad in ("wrong_run", "wrong_packet", "wrong_attestation", "wrong_score"):
            overrides = {
                "wrong_run": {"runner_execution_id_sha256": "1" * 64},
                "wrong_packet": {"packet_sha256": "2" * 64},
                "wrong_attestation": {"execution_attestation_sha256": "3" * 64},
                "wrong_score": {"score_archive_sha256": "4" * 64},
            }
            with self.subTest(bad=bad):
                self.sign(**overrides[bad])
                with self.assertRaisesRegex(ValueError, "HOLD"):
                    self.consume()

    def test_bogus_signed_authorization_or_operator_root_denied(self):
        self.sign()
        fake = json.loads(self.auth_path.read_text())
        fake["signature_hex"] = "00" * 64
        self.auth_path.write_text(json.dumps(fake))
        with self.assertRaisesRegex(ValueError, "HOLD"):
            self.consume()
        self.sign()
        with patch.object(run, "APPROVED_OPERATOR_PUBLIC_KEY_SHA256", "f" * 64):
            with patch.object(run, "APPROVED_OPERATOR_CLAIM_LEDGER_PATH",
                              str(self.state_path)):
                with self.assertRaisesRegex(ValueError, "HOLD"):
                    run.consume_unique_run(
                        self.auth_path, self.scorer_private,
                        provenance=self.provenance, packet_sha256=self.packet)

    def test_scorer_same_as_operator_denied(self):
        self.sign()
        same = dict(self.provenance)
        same["runner_public_key_sha256"] = self.pubsha
        with self.assertRaisesRegex(ValueError, "HOLD"):
            self.consume(provenance=same)

    def test_noninteger_sequences_or_missing_fields_denied(self):
        for bad in (True, 1.0, -1, None):
            with self.subTest(bad=bad):
                self.sign(sequence=bad)
                with self.assertRaisesRegex(ValueError, "HOLD"):
                    self.consume()
        self.sign()
        v = json.loads(self.auth_path.read_text())
        v["private_passage"] = "FORBIDDEN"
        self.auth_path.write_text(json.dumps(v))
        with self.assertRaisesRegex(ValueError, "HOLD"):
            self.consume()

    def test_no_operator_owned_journal_denied(self):
        self.sign()
        self.state_path.unlink()
        with self.assertRaisesRegex(ValueError, "HOLD"):
            self.consume()

    def test_operator_state_mode_and_duplicate_keys_denied(self):
        self.sign()
        os.chmod(self.state_path, 0o644)
        with self.assertRaisesRegex(ValueError, "HOLD"):
            self.consume()
        self.create_state()
        self.state_path.write_text(
            '{"schema":"' + run.STATE_SCHEMA +
            '","schema":"' + run.STATE_SCHEMA +
            '","last_sequence":0,"last_claim_sha256":"' + "0" * 64 + '"}')
        with self.assertRaisesRegex(ValueError, "HOLD"):
            self.consume()

    def test_corrupt_journal_does_not_autorepair(self):
        self.sign()
        self.state_path.write_text('{"interrupted":')
        with self.assertRaisesRegex(ValueError, "HOLD"):
            self.consume()
        self.assertEqual(self.state_path.read_text(), '{"interrupted":')

    def test_world_writable_operator_directory_denied(self):
        self.sign()
        os.chmod(self.operator_dir, 0o777)
        with self.assertRaisesRegex(ValueError, "HOLD"):
            self.consume()

    def test_symlink_or_hardlinked_operator_state_denied(self):
        self.sign()
        alias = self.operator_dir / "operator-state-alias.json"
        alias.symlink_to(self.state_path)
        original = self.state_path
        self.state_path = alias
        try:
            with self.assertRaisesRegex(ValueError, "HOLD"):
                self.consume()
        finally:
            self.state_path = original
        os.link(self.state_path, self.operator_dir / "extra-hardlink")
        with self.assertRaisesRegex(ValueError, "HOLD"):
            self.consume()

    def test_preexisting_run_marker_denies_advance(self):
        self.sign()
        marker = self.state_path.with_name(
            self.state_path.name + ".used-" + self.provenance["execution_id_sha256"])
        marker.write_text("previously used fixture marker")
        os.chmod(marker, 0o600)
        with self.assertRaisesRegex(ValueError, "HOLD"):
            self.consume()
        self.assertEqual(json.loads(self.state_path.read_text())["last_sequence"], 0)

    def test_corrupt_marker_does_not_get_deleted(self):
        self.sign()
        marker = self.state_path.with_name(
            self.state_path.name + ".used-" + self.provenance["execution_id_sha256"])
        marker.write_text("CORRUPT_PRIVATE_MARKER")
        with self.assertRaisesRegex(ValueError, "HOLD"):
            self.consume()
        self.assertEqual(marker.read_text(), "CORRUPT_PRIVATE_MARKER")

    def test_fresh_nonce_and_signed_receipt_enforced_before_local_mutation(self):
        for fault in ("stale_nonce", "wrong_counter", "boolean_counter",
                      "not_committed", "wrong_execution", "extra_field",
                      "tampered_signature", "wrong_public_key", "offline"):
            with self.subTest(fault=fault):
                self.create_state()
                self.highwater = (0, "0" * 64)
                self.witness_fault = fault
                self.sign()
                with self.assertRaisesRegex(ValueError, "HOLD"):
                    self.consume()
                self.assertEqual(
                    json.loads(self.state_path.read_text())["last_sequence"], 0)
                marker = self.state_path.with_name(
                    self.state_path.name + ".used-" +
                    self.provenance["execution_id_sha256"])
                self.assertFalse(marker.exists())
        self.witness_fault = None

    def test_restored_local_state_and_deleted_marker_cannot_replay_remote(self):
        self.sign()
        accepted = self.consume()
        self.assertEqual(accepted["external_highwater_sequence"], 1)
        self.assertEqual(self.highwater[0], 1)
        marker = self.state_path.with_name(
            self.state_path.name + ".used-" +
            self.provenance["execution_id_sha256"])
        marker.unlink()  # simulated privileged rollback of LOCAL state only
        self.create_state()  # local claim journal restored to prior snapshot
        self.sign()
        with self.assertRaisesRegex(ValueError, "HOLD"):
            self.consume()
        self.assertEqual(
            json.loads(self.state_path.read_text())["last_sequence"], 0)
        self.assertEqual(self.highwater[0], 1)

    def test_remote_commit_precedes_local_marker_and_claim(self):
        self.sign()
        self.witness_fault = "not_committed"
        with self.assertRaisesRegex(ValueError, "HOLD"):
            self.consume()
        self.assertEqual(self.highwater[0], 1)
        self.assertEqual(
            json.loads(self.state_path.read_text())["last_sequence"], 0)
        self.assertFalse(list(self.operator_dir.glob("*.used-*")))

    def test_sqlite_fixture_end_to_end_local_rollback_rejected(self):
        # The database is still local TEST-ONLY; this validates interfaces,
        # not a real independent trust domain or remote durable evidence.
        sys.path.insert(0, str(R.parents[0] / "tests" / "fixtures"))
        from exp053_sqlite_cas_fixture import SQLiteCASFixture
        sqlite = SQLiteCASFixture.bootstrap(
            Path(self.tmp.name) / "fixture-witness-only.sqlite",
            self.witness_key, self.pubsha)
        self.synthetic_external_cas = sqlite.cas
        self.sign()
        claim = self.consume()
        self.assertEqual(claim["external_highwater_sequence"], 1)
        self.assertEqual(sqlite.highwater(), (1, claim["operator_authorization_sha256"], 1))
        # Simulated privileged rollback of ONLY the local journal and marker.
        marker = self.state_path.with_name(
            self.state_path.name + ".used-" + self.provenance["execution_id_sha256"])
        marker.unlink()
        self.create_state()
        with self.assertRaisesRegex(ValueError, "HOLD"):
            self.consume()
        self.assertEqual(sqlite.highwater(), (1, claim["operator_authorization_sha256"], 1))
        self.assertEqual(json.loads(self.state_path.read_text())["last_sequence"], 0)

    def test_unconfigured_external_authorities_always_hold(self):
        self.sign()
        with patch.object(run, "APPROVED_OPERATOR_PUBLIC_KEY_SHA256",
                          self.pubsha), patch.object(
            run, "APPROVED_OPERATOR_CLAIM_LEDGER_PATH",
            str(self.state_path)):
            with self.assertRaisesRegex(ValueError, "HOLD"):
                run.consume_unique_run(
                    self.auth_path, self.scorer_private,
                    provenance=self.provenance, packet_sha256=self.packet)
        self.assertEqual(self.highwater[0], 0)

    def test_state_inside_scorer_private_directory_denied(self):
        self.sign()
        untrusted = self.scorer_private / "untrusted-operator.json"
        untrusted.write_bytes(self.state_path.read_bytes())
        os.chmod(untrusted, 0o600)
        with patch.object(run, "APPROVED_OPERATOR_PUBLIC_KEY_SHA256",
                          self.pubsha), patch.object(
            run, "APPROVED_OPERATOR_CLAIM_LEDGER_PATH", str(untrusted)):
            with self.assertRaisesRegex(ValueError, "HOLD"):
                run.consume_unique_run(
                    self.auth_path, self.scorer_private,
                    provenance=self.provenance, packet_sha256=self.packet)

    def test_synthetic_concurrent_consumers_only_one_wins(self):
        self.sign()
        # All mock roots belong to ONE outer scope. Nested patch.object from
        # racing threads can restore a nonempty value after both finish.
        with patch.object(run, "APPROVED_OPERATOR_PUBLIC_KEY_SHA256",
                          self.pubsha), patch.object(
            run, "APPROVED_OPERATOR_CLAIM_LEDGER_PATH", str(self.state_path)
        ), patch.object(
            witness, "APPROVED_EXTERNAL_WITNESS_PUBLIC_KEY_SHA256",
            self.witness_pub_sha
        ), patch.object(
            witness, "APPROVED_EXTERNAL_WITNESS_ADAPTER_SHA256", "e" * 64
        ), patch.object(
            witness, "APPROVED_WITNESS_OPERATOR_PUBLIC_KEY_SHA256", self.pubsha
        ), patch.object(witness, "_external_cas", self.synthetic_external_cas):
            with ThreadPoolExecutor(max_workers=2) as pool:
                results = list(pool.map(lambda _: self._try_consume(), range(2)))
        self.assertEqual(sum(value == "success" for value in results), 1)
        self.assertEqual(sum(value == "hold" for value in results), 1)
        self.assertEqual(json.loads(self.state_path.read_text())["last_sequence"], 1)
        self.assertEqual(run.APPROVED_OPERATOR_PUBLIC_KEY_SHA256, "")
        self.assertEqual(witness.APPROVED_EXTERNAL_WITNESS_PUBLIC_KEY_SHA256, "")
        self.assertEqual(witness.APPROVED_WITNESS_OPERATOR_PUBLIC_KEY_SHA256, "")

    def _try_consume(self):
        try:
            run.consume_unique_run(
                self.auth_path, self.scorer_private,
                provenance=self.provenance, packet_sha256=self.packet)
            return "success"
        except ValueError as exc:
            if "HOLD" in str(exc):
                return "hold"
            raise

if __name__ == "__main__":
    unittest.main()
