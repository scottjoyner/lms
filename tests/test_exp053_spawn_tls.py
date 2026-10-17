"""Separate spawned-OS-process TLS witness acceptance: synthetic ONLY."""
import base64
import hashlib
import json
import os
import ssl
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
sys.path.insert(0, str(ROOT / "tests" / "fixtures"))

import exp053_external_witness as witness
from exp053_mtls_loopback_fixture import LoopbackMTLSWitness
from exp053_spawn_tls_fixture import SpawnedTLSWitness
from exp053_sqlite_cas_fixture import SQLiteCASFixture
from exp053_race_review import canonical

try:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
except ImportError:
    Ed25519PrivateKey = None
    serialization = None


def sha(value):
    return hashlib.sha256(value).hexdigest()


class ProtocolFreeze(unittest.TestCase):
    def test_frozen_spec_and_no_test_import_in_production(self):
        file = ROOT / "research" / "EXP053-PROCESS-ISOLATED-MTLS-SPEC-2026-10-17.md"
        self.assertEqual(sha(file.read_bytes()),
                         "eff7e93d3aa4744c4027c2171f93d301bacb172c5daa4cb3316839d09c325c3b")
        source = (ROOT / "research" / "exp053_external_witness.py").read_text()
        self.assertNotIn("exp053_spawn_tls_fixture", source)
        self.assertNotIn("ssl.", source)
        self.assertEqual(witness.APPROVED_EXTERNAL_WITNESS_ADAPTER_SHA256, "")


@unittest.skipUnless(Ed25519PrivateKey is not None, "cryptography optional on CI")
class SpawnedWitnessAcceptance(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.operator_key = Ed25519PrivateKey.generate()
        self.witness_key = Ed25519PrivateKey.generate()
        self.operator_pub = self.operator_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw)
        self.witness_pub = self.witness_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw)
        self.operator_sha = sha(self.operator_pub)
        self.witness_sha = sha(self.witness_pub)
        self.dbfile = self.root / "synthetic-only.sqlite"
        self.db = SQLiteCASFixture.bootstrap(
            self.dbfile, self.witness_key, self.operator_sha)
        self.front = LoopbackMTLSWitness(self.root, self.db)

    def new_server(self, *, failure=None):
        return SpawnedTLSWitness(self.front, self.dbfile, self.witness_key,
                                 self.operator_sha, failure=failure)

    def request(self, seq=1, prior=None, run_id="a" * 64):
        prev = self.db.highwater()[1] if prior is None else prior
        auth = {
            "schema": "exp053-operator-run-authorization-v1",
            "runner_execution_id_sha256": run_id,
            "execution_attestation_sha256": "b" * 64,
            "packet_sha256": "c" * 64,
            "score_archive_sha256": "d" * 64,
            "operator_custody_reference_sha256": "e" * 64,
            "sequence": seq,
            "previous_claim_sha256": prev,
            "operator_public_key_b64": base64.b64encode(self.operator_pub).decode(),
        }
        auth["signature_hex"] = self.operator_key.sign(canonical(auth)).hex()
        raw = json.dumps(auth, sort_keys=True).encode() + b"\n"
        return {
            "schema": witness.REQUEST_SCHEMA,
            "nonce_hex": os.urandom(32).hex(),
            "sequence": seq,
            "expected_previous_sequence": seq - 1,
            "previous_claim_sha256": prev,
            "claim_sha256": sha(raw),
            "runner_execution_id_sha256": run_id,
            "operator_authorization_b64": base64.b64encode(raw).decode(),
        }, raw

    def test_child_pid_and_fresh_db_commit_survives_process_exit(self):
        req, _ = self.request()
        with self.new_server() as server:
            self.assertNotEqual(server.pid, os.getpid())
            response = server.exchange(req)
            self.assertEqual(response["status"], "committed")
            self.assertEqual(response["witness_counter"], 1)
        self.assertEqual(self.db.highwater(), (1, req["claim_sha256"], 1))
        self.assertFalse(server.process.is_alive())

    def test_real_source_client_response_validator_over_spawned_tls(self):
        req, raw = self.request()
        with self.new_server() as child:
            with patch.object(
                witness, "APPROVED_EXTERNAL_WITNESS_PUBLIC_KEY_SHA256",
                self.witness_sha), patch.object(
                witness, "APPROVED_EXTERNAL_WITNESS_ADAPTER_SHA256", "f"*64
            ), patch.object(
                witness, "APPROVED_WITNESS_OPERATOR_PUBLIC_KEY_SHA256",
                self.operator_sha), patch.object(
                witness, "_external_cas", child.exchange):
                result = witness.commit_independent_highwater(
                    sequence=1, previous_claim_sha256="0" * 64,
                    claim_sha256=req["claim_sha256"],
                    execution_id_sha256=req["runner_execution_id_sha256"],
                    operator_public_key_sha256=self.operator_sha,
                    scorer_public_key_sha256="2" * 64,
                    operator_authorization_bytes=raw)
            self.assertEqual(result["external_highwater_sequence"], 1)
        self.assertEqual(self.db.highwater()[0], 1)

    def test_duplicate_claim_rejected_after_full_server_restart(self):
        req, _ = self.request()
        with self.new_server() as child:
            child.exchange(req)
        with self.new_server() as restarted:
            with self.assertRaises((OSError, ValueError, ConnectionError)):
                restarted.exchange(req)
        self.assertEqual(self.db.highwater(), (1, req["claim_sha256"], 1))

    def test_precommit_process_failure_does_not_consume_claim(self):
        req, _ = self.request()
        with self.new_server(failure="before_commit") as child:
            with self.assertRaises((OSError, ValueError, ConnectionError)):
                child.exchange(req)
        self.assertEqual(self.db.highwater(), (0, "0" * 64, 0))
        with self.new_server() as restarted:
            self.assertEqual(restarted.exchange(req)["witness_counter"], 1)
        self.assertEqual(self.db.highwater()[0], 1)

    def test_postcommit_child_crash_is_durable_and_replay_refused(self):
        req, _ = self.request()
        with self.new_server(failure="after_commit") as child:
            with self.assertRaises((OSError, ValueError, ConnectionError)):
                child.exchange(req)
        self.assertEqual(child.process.exitcode, 73)
        self.assertEqual(self.db.highwater(), (1, req["claim_sha256"], 1))
        with self.new_server() as restarted:
            with self.assertRaises((OSError, ValueError, ConnectionError)):
                restarted.exchange(req)
        self.assertEqual(self.db.highwater()[0], 1)

    def test_invalid_same_ca_client_rejected_in_separate_process(self):
        req, _ = self.request()
        with self.new_server() as child:
            with self.assertRaises((OSError, ValueError, ConnectionError)):
                child.exchange(req, client="other")
        self.assertEqual(self.db.highwater(), (0, "0" * 64, 0))

    def test_missing_tls_client_identity_denied_in_child(self):
        req, _ = self.request()
        with self.new_server() as child:
            with self.assertRaises((OSError, ssl.SSLError, ValueError)):
                child.exchange(req, client="none")
        self.assertEqual(self.db.highwater(), (0, "0" * 64, 0))

    def test_untrusted_server_ca_or_bad_hostname_never_mutate(self):
        req, _ = self.request()
        with self.new_server() as child:
            with self.assertRaises((OSError, ValueError, ConnectionError)):
                child.exchange(req, hostname="wrong.example")
        with self.new_server() as child:
            with self.assertRaises((OSError, ValueError, ConnectionError)):
                child.exchange(req, trusted_ca=self.front.untrusted_ca.ca_pem)
        self.assertEqual(self.db.highwater(), (0, "0" * 64, 0))

    def test_bad_signed_operator_authorization_rejected(self):
        req, _ = self.request()
        signed = json.loads(base64.b64decode(req["operator_authorization_b64"]))
        signed["signature_hex"] = "00" * 64
        raw = json.dumps(signed, sort_keys=True).encode()
        req["operator_authorization_b64"] = base64.b64encode(raw).decode()
        req["claim_sha256"] = sha(raw)
        with self.new_server() as child:
            with self.assertRaises((OSError, ValueError, ConnectionError)):
                child.exchange(req)
        self.assertEqual(self.db.highwater(), (0, "0" * 64, 0))

    def test_two_independent_witness_processes_share_atomic_sqlite(self):
        first_req, _ = self.request()
        second_req, _ = self.request()
        with self.new_server() as first, self.new_server() as second:
            with ThreadPoolExecutor(max_workers=2) as pool:
                future1 = pool.submit(first.exchange, first_req)
                future2 = pool.submit(second.exchange, second_req)
                results = []
                for future in (future1, future2):
                    try:
                        results.append(future.result(timeout=4)["status"])
                    except (OSError, ValueError, ConnectionError):
                        results.append("denied")
        self.assertEqual(results.count("committed"), 1)
        self.assertEqual(results.count("denied"), 1)
        self.assertEqual(self.db.highwater()[0], 1)
        self.assertEqual(self.db.highwater()[2], 1)


if __name__ == "__main__":
    unittest.main()
