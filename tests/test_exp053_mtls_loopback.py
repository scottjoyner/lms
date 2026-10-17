"""EXP-053 disposable mutual-TLS loopback evidence, never a live witness."""
import base64
import hashlib
import json
import os
import ssl
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
sys.path.insert(0, str(ROOT / "tests" / "fixtures"))

import exp053_external_witness as witness
from exp053_race_review import canonical
from exp053_sqlite_cas_fixture import SQLiteCASFixture
from exp053_mtls_loopback_fixture import LoopbackMTLSWitness

try:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
except ImportError:
    Ed25519PrivateKey = None
    serialization = None


def sha(data):
    return hashlib.sha256(data).hexdigest()


class FreezeAndNoProductionAdapter(unittest.TestCase):
    def test_frozen_loopback_spec_and_no_live_adapter(self):
        p = ROOT / "research" / "EXP053-LOOPBACK-MTLS-WITNESS-SPEC-2026-10-17.md"
        self.assertEqual(sha(p.read_bytes()),
                         "67de58c79ec9b99de588a022a152b21fcdcd00461fbd9111d88e4fc50e9fb87d")
        self.assertEqual(witness.APPROVED_EXTERNAL_WITNESS_ADAPTER_SHA256, "")
        self.assertEqual(witness.APPROVED_EXTERNAL_WITNESS_PUBLIC_KEY_SHA256, "")
        self.assertEqual(witness.APPROVED_WITNESS_OPERATOR_PUBLIC_KEY_SHA256, "")
        production = (ROOT / "research" / "exp053_external_witness.py").read_text()
        self.assertNotIn("exp053_mtls_loopback_fixture", production)
        self.assertNotIn("ssl.", production)
        self.assertNotIn("socket.", production)


@unittest.skipUnless(Ed25519PrivateKey is not None, "cryptography optional in hosted CI")
class TestOnlyMutualTLS(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.operator = Ed25519PrivateKey.generate()
        self.witness_signing = Ed25519PrivateKey.generate()
        self.operator_raw = self.operator.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw)
        self.witness_raw = self.witness_signing.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw)
        self.operator_sha = sha(self.operator_raw)
        self.witness_sha = sha(self.witness_raw)
        self.db = SQLiteCASFixture.bootstrap(
            self.root / "disposable-witness.sqlite", self.witness_signing,
            self.operator_sha)
        self.server = LoopbackMTLSWitness(self.root, self.db)

    def authorization(self, sequence=1, previous=None, execution="a" * 64):
        if previous is None:
            previous = self.db.highwater()[1]
        doc = {
            "schema": "exp053-operator-run-authorization-v1",
            "runner_execution_id_sha256": execution,
            "execution_attestation_sha256": "b" * 64,
            "packet_sha256": "c" * 64,
            "score_archive_sha256": "d" * 64,
            "operator_custody_reference_sha256": "e" * 64,
            "sequence": sequence,
            "previous_claim_sha256": previous,
            "operator_public_key_b64": base64.b64encode(self.operator_raw).decode(),
        }
        doc["signature_hex"] = self.operator.sign(canonical(doc)).hex()
        raw = json.dumps(doc, sort_keys=True).encode() + b"\n"
        req = {
            "schema": witness.REQUEST_SCHEMA,
            "nonce_hex": os.urandom(32).hex(),
            "sequence": sequence,
            "expected_previous_sequence": sequence - 1,
            "previous_claim_sha256": previous,
            "claim_sha256": sha(raw),
            "runner_execution_id_sha256": execution,
            "operator_authorization_b64": base64.b64encode(raw).decode(),
        }
        return req, raw

    def test_valid_mutual_tls13_and_signed_cas(self):
        req, _ = self.authorization()
        reply = self.server.exchange(req)
        self.assertEqual(reply["status"], "committed")
        self.assertEqual(reply["witness_counter"], 1)
        self.assertEqual(self.db.highwater(), (1, req["claim_sha256"], 1))

    def test_real_client_validator_accepts_only_pinned_test_receipt(self):
        req, raw = self.authorization()
        with patch.object(witness, "APPROVED_EXTERNAL_WITNESS_PUBLIC_KEY_SHA256",
                          self.witness_sha), patch.object(
            witness, "APPROVED_EXTERNAL_WITNESS_ADAPTER_SHA256", "f" * 64), patch.object(
            witness, "APPROVED_WITNESS_OPERATOR_PUBLIC_KEY_SHA256",
            self.operator_sha), patch.object(
            witness, "_external_cas", self.server.exchange):
            result = witness.commit_independent_highwater(
                sequence=1, previous_claim_sha256="0" * 64,
                claim_sha256=req["claim_sha256"],
                execution_id_sha256="a" * 64,
                operator_public_key_sha256=self.operator_sha,
                scorer_public_key_sha256="2" * 64,
                operator_authorization_bytes=raw)
        self.assertEqual(result["external_highwater_sequence"], 1)
        self.assertEqual(self.db.highwater()[0], 1)

    def test_missing_client_certificate_rejected_before_cas(self):
        req, _ = self.authorization()
        with self.assertRaises((OSError, ssl.SSLError, ConnectionError, ValueError)):
            self.server.exchange(req, client="none")
        self.assertEqual(self.db.highwater(), (0, "0" * 64, 0))

    def test_wrong_but_ca_issued_client_leaf_rejected(self):
        req, _ = self.authorization()
        with self.assertRaises((OSError, ssl.SSLError, ConnectionError, ValueError)):
            self.server.exchange(req, client="other")
        self.assertEqual(self.db.highwater(), (0, "0" * 64, 0))

    def test_client_certificate_untrusted_ca_rejected(self):
        req, _ = self.authorization()
        with self.assertRaises((OSError, ssl.SSLError, ConnectionError, ValueError)):
            self.server.exchange(req, client="untrusted")
        self.assertEqual(self.db.highwater(), (0, "0" * 64, 0))

    def test_wrong_server_hostname_rejected(self):
        req, _ = self.authorization()
        with self.assertRaises((OSError, ssl.SSLError, ConnectionError, ValueError)):
            self.server.exchange(req, hostname="evil.example")
        self.assertEqual(self.db.highwater(), (0, "0" * 64, 0))

    def test_server_ca_not_trusted_rejected(self):
        req, _ = self.authorization()
        with self.assertRaises((OSError, ssl.SSLError, ConnectionError, ValueError)):
            self.server.exchange(req, client_trust_ca=self.server.untrusted_ca.ca_pem)
        self.assertEqual(self.db.highwater(), (0, "0" * 64, 0))

    def test_wrong_operator_signature_denied_after_tls(self):
        req, _ = self.authorization()
        auth = json.loads(base64.b64decode(req["operator_authorization_b64"]))
        auth["signature_hex"] = "00" * 64
        changed = json.dumps(auth, sort_keys=True).encode()
        req["operator_authorization_b64"] = base64.b64encode(changed).decode()
        req["claim_sha256"] = sha(changed)
        with self.assertRaises((OSError, ConnectionError, ValueError)):
            self.server.exchange(req)
        self.assertEqual(self.db.highwater(), (0, "0" * 64, 0))

    def test_ambiguous_operator_json_denied_after_tls(self):
        req, _ = self.authorization()
        raw = base64.b64decode(req["operator_authorization_b64"])
        malformed = raw.rstrip()[:-1] + b',"schema":"exp053-operator-run-authorization-v1"}'
        req["operator_authorization_b64"] = base64.b64encode(malformed).decode()
        req["claim_sha256"] = sha(malformed)
        with self.assertRaises((OSError, ConnectionError, ValueError)):
            self.server.exchange(req)
        self.assertEqual(self.db.highwater(), (0, "0" * 64, 0))

    def test_oversized_transport_payload_refused(self):
        raw = b"x" * (32 * 1024 + 1)
        with self.assertRaises((OSError, ConnectionError, ValueError)):
            self.server.exchange(raw)
        self.assertEqual(self.db.highwater(), (0, "0" * 64, 0))

    def test_replay_second_tls_connection_cannot_repeat_claim(self):
        req, _ = self.authorization()
        self.server.exchange(req)
        with self.assertRaises((OSError, ConnectionError, ValueError)):
            self.server.exchange(req)
        self.assertEqual(self.db.highwater(), (1, req["claim_sha256"], 1))

    def test_no_certificate_private_keys_in_repo(self):
        self.assertTrue(all(p.is_file() and (p.stat().st_mode & 0o777) == 0o600
                            for p in self.server.ca.directory.glob("*.key.pem")))
        production = ROOT / "research" / "exp053_external_witness.py"
        self.assertNotIn("server.key.pem", production.read_text())


if __name__ == "__main__":
    unittest.main()
