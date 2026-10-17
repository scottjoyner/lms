"""Offline durability/concurrency checks for a NONDEPLOYABLE SQLite fixture."""
import hashlib
import multiprocessing
import os
import sqlite3
import subprocess
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
from exp053_sqlite_cas_fixture import SQLiteCASFixture

try:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
except ImportError:
    Ed25519PrivateKey = None
    serialization = None


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


class FrozenFixtureContract(unittest.TestCase):
    def test_frozen_synthetic_fixture_protocol(self):
        path = ROOT / "research" / "EXP053-SQLITE-WITNESS-FIXTURE-SPEC-2026-10-17.md"
        self.assertEqual(witness.APPROVED_EXTERNAL_WITNESS_ADAPTER_SHA256, "")
        self.assertEqual(witness.APPROVED_EXTERNAL_WITNESS_PUBLIC_KEY_SHA256, "")
        self.assertEqual(sha(path.read_bytes()),
                         "1d4589ba32c26ab6abc5b5f8e3659f1e863dbd69a8ce443c1b1400ede426209a")

    def test_no_production_adapter_or_endpoint(self):
        self.assertIn("tests/fixtures", str(ROOT / "tests/fixtures"))
        self.assertNotIn("exp053_sqlite_cas_fixture", (
            ROOT / "research" / "exp053_external_witness.py").read_text())
        self.assertNotIn("sqlite3", (
            ROOT / "research" / "exp053_external_witness.py").read_text())


@unittest.skipUnless(Ed25519PrivateKey is not None, "cryptography optional in hosted CI")
class DurableSyntheticWitness(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.path = self.root / "synthetic-not-a-service.sqlite"
        self.key = Ed25519PrivateKey.generate()
        self.fixture = SQLiteCASFixture.bootstrap(self.path, self.key)
        public = self.key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw)
        self.witness_fingerprint = sha(public)
        self.operator = "1" * 64
        self.scorer = "2" * 64
        self.previous = "0" * 64

    def req(self, sequence=1, previous=None, claim=None, execution=None,
            nonce=None):
        return {
            "schema": witness.REQUEST_SCHEMA,
            "nonce_hex": nonce or os.urandom(32).hex(),
            "sequence": sequence,
            "expected_previous_sequence": sequence - 1,
            "previous_claim_sha256": previous if previous is not None else self.previous,
            "claim_sha256": claim if claim is not None else str(sequence) * 64,
            "runner_execution_id_sha256":
                execution if execution is not None else "a" * 64,
        }

    def checked_commit(self, seq=1, previous=None, claim=None, execution=None):
        with patch.object(
            witness, "APPROVED_EXTERNAL_WITNESS_PUBLIC_KEY_SHA256",
            self.witness_fingerprint), patch.object(
            witness, "APPROVED_EXTERNAL_WITNESS_ADAPTER_SHA256",
            "b" * 64), patch.object(
            witness, "_external_cas", self.fixture.cas):
            return witness.commit_independent_highwater(
                sequence=seq,
                previous_claim_sha256=previous if previous is not None else self.previous,
                claim_sha256=claim if claim is not None else str(seq) * 64,
                execution_id_sha256=execution if execution is not None else "a" * 64,
                operator_public_key_sha256=self.operator,
                scorer_public_key_sha256=self.scorer)

    def test_genesis_and_reopen_new_connection(self):
        self.assertEqual(self.fixture.highwater(), (0, self.previous, 0))
        result = self.checked_commit()
        self.assertEqual(result["external_highwater_sequence"], 1)
        self.assertEqual(self.fixture.highwater(), (1, "1" * 64, 1))
        second = SQLiteCASFixture(self.path, self.key)
        self.assertEqual(second.highwater(), (1, "1" * 64, 1))
        self.assertEqual(self.path.stat().st_mode & 0o777, 0o600)

    def test_multiple_independent_highwater_sequences(self):
        self.checked_commit()
        result = self.checked_commit(
            seq=2, previous="1" * 64, execution="b" * 64)
        self.assertEqual(result["external_highwater_sequence"], 2)
        self.assertEqual(self.fixture.highwater(), (2, "2" * 64, 2))

    def test_duplicate_execution_id_rejected_with_fresh_sequence(self):
        self.checked_commit()
        # This has an entirely new operator claim but same original execution ID.
        with self.assertRaisesRegex(ValueError, "HOLD"):
            self.checked_commit(seq=2, previous="1" * 64, execution="a" * 64)
        self.assertEqual(self.fixture.highwater(), (1, "1" * 64, 1))

    def test_reused_nonce_and_claim_digest_denied(self):
        request1 = self.req(nonce="c" * 64)
        self.fixture.cas(request1)
        for changed in (
            self.req(sequence=2, previous="1" * 64,
                     claim="2" * 64, execution="b" * 64, nonce="c" * 64),
            self.req(sequence=2, previous="1" * 64,
                     claim="1" * 64, execution="b" * 64),
        ):
            with self.assertRaises(sqlite3.IntegrityError):
                self.fixture.cas(changed)
            self.assertEqual(self.fixture.highwater(), (1, "1" * 64, 1))

    def test_stale_or_forked_history_rejected(self):
        self.checked_commit()
        for kw in ({"seq": 1}, {"seq": 3, "previous": "1" * 64},
                   {"seq": 2, "previous": "f" * 64}):
            with self.assertRaisesRegex(ValueError, "HOLD"):
                self.checked_commit(**kw)
        self.assertEqual(self.fixture.highwater(), (1, "1" * 64, 1))

    def test_malformed_request_type_denied_before_write(self):
        request = self.req()
        for field, value in (
            ("sequence", True), ("sequence", 1.0),
            ("nonce_hex", "bad"), ("runner_execution_id_sha256", ["a"]),
            ("expected_previous_sequence", True),
        ):
            old = request[field]
            request[field] = value
            with self.assertRaises(ValueError):
                self.fixture.cas(request)
            self.assertEqual(self.fixture.highwater(), (0, self.previous, 0))
            request[field] = old

    def test_interruption_before_commit_rolls_back_atomically(self):
        self.fixture.fail_before_commit = True
        with self.assertRaises(RuntimeError):
            self.fixture.cas(self.req())
        self.assertEqual(self.fixture.highwater(), (0, self.previous, 0))
        self.fixture.fail_before_commit = False
        self.checked_commit()
        self.assertEqual(self.fixture.highwater(), (1, "1" * 64, 1))

    def test_receipt_lost_after_commit_does_not_uncommit(self):
        self.fixture.fail_after_commit = True
        with self.assertRaisesRegex(ValueError, "HOLD"):
            self.checked_commit()
        self.assertEqual(self.fixture.highwater(), (1, "1" * 64, 1))
        self.fixture.fail_after_commit = False
        with self.assertRaisesRegex(ValueError, "HOLD"):
            self.checked_commit()

    def test_two_racing_connections_one_success(self):
        requests = [self.req(nonce="a" * 64), self.req(nonce="b" * 64)]
        separate = [SQLiteCASFixture(self.path, self.key) for _ in range(2)]
        with ThreadPoolExecutor(max_workers=2) as pool:
            outcomes = list(pool.map(
                lambda item: attempt(item[0].cas, item[1]),
                zip(separate, requests)))
        self.assertEqual(outcomes.count(True), 1)
        self.assertEqual(outcomes.count(False), 1)
        self.assertEqual(self.fixture.highwater()[0], 1)

    def test_reopened_file_visible_to_new_process(self):
        self.checked_commit()
        script = (
            "import sqlite3,sys; c=sqlite3.connect(sys.argv[1]); "
            "print(c.execute('SELECT sequence FROM witness_state').fetchone()[0])"
        )
        out = subprocess.check_output(
            [sys.executable, "-c", script, str(self.path)], text=True).strip()
        self.assertEqual(out, "1")

    @unittest.skipUnless("spawn" in multiprocessing.get_all_start_methods(),
                         "spawn-process test only on supported Python hosts")
    def test_two_independent_processes_compete_for_same_db(self):
        context = multiprocessing.get_context("spawn")
        queue = context.Queue()
        req1 = self.req(nonce="5" * 64)
        req2 = self.req(nonce="6" * 64)
        seed = self.key.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption())
        processes = [
            context.Process(target=_process_attempt,
                            args=(str(self.path), seed, req, queue))
            for req in (req1, req2)]
        for p in processes:
            p.start()
        for p in processes:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
                p.join()
                self.fail("Synthetic SQLite CAS subprocess hung")
            self.assertEqual(p.exitcode, 0)
        result = [queue.get(timeout=2) for _ in processes]
        self.assertEqual(result.count(True), 1)
        self.assertEqual(result.count(False), 1)
        self.assertEqual(self.fixture.highwater(), (1, "1" * 64, 1))

    def test_absent_real_pins_remain_unconfigured(self):
        self.assertEqual(witness.APPROVED_EXTERNAL_WITNESS_PUBLIC_KEY_SHA256, "")
        self.assertEqual(witness.APPROVED_EXTERNAL_WITNESS_ADAPTER_SHA256, "")
        with self.assertRaisesRegex(ValueError, "HOLD"):
            witness.commit_independent_highwater(
                sequence=1, previous_claim_sha256=self.previous,
                claim_sha256="1" * 64, execution_id_sha256="a" * 64,
                operator_public_key_sha256=self.operator,
                scorer_public_key_sha256=self.scorer)


def attempt(fn, request):
    try:
        fn(request)
        return True
    except (ValueError, sqlite3.IntegrityError):
        return False


def _process_attempt(path, test_only_private_seed, request, queue):
    # Disposable synthetic seed only; NEVER a real witness key.
    key = Ed25519PrivateKey.from_private_bytes(test_only_private_seed)
    queue.put(attempt(SQLiteCASFixture(path, key).cas, request))


if __name__ == "__main__":
    unittest.main()
