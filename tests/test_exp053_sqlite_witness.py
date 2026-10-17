"""Isolated authenticated SQLite witness tests. No real trust or model execution."""
import base64
import hashlib
import json
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
from exp053_race_review import canonical
from exp053_witness_request_auth import verify_authenticated_request

try:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
except ImportError:
    Ed25519PrivateKey = None
    serialization = None


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


class FrozenFixtureContract(unittest.TestCase):
    def test_frozen_synthetic_fixture_protocols(self):
        path = ROOT / "research" / "EXP053-SQLITE-WITNESS-FIXTURE-SPEC-2026-10-17.md"
        auth = ROOT / "research" / "EXP053-AUTHENTICATED-WITNESS-REQUEST-SPEC-2026-10-17.md"
        self.assertEqual(sha(path.read_bytes()),
                         "1d4589ba32c26ab6abc5b5f8e3659f1e863dbd69a8ce443c1b1400ede426209a")
        self.assertEqual(sha(auth.read_bytes()),
                         "8bc1c1c03ae5672c86e8df5dc1112e53a233902c32f2811fcaceb237f7f299a8")
        self.assertEqual(witness.APPROVED_EXTERNAL_WITNESS_ADAPTER_SHA256, "")
        self.assertEqual(witness.APPROVED_EXTERNAL_WITNESS_PUBLIC_KEY_SHA256, "")
        self.assertEqual(witness.APPROVED_WITNESS_OPERATOR_PUBLIC_KEY_SHA256, "")

    def test_no_production_fixture_import(self):
        code = (ROOT / "research" / "exp053_external_witness.py").read_text()
        self.assertNotIn("exp053_sqlite_cas_fixture", code)
        self.assertNotIn("sqlite3", code)


@unittest.skipUnless(Ed25519PrivateKey is not None, "crypto optional on hosted CI")
class AuthenticatedDurableFixture(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.path = self.root / "synthetic-not-a-service.sqlite"
        self.witness_key = Ed25519PrivateKey.generate()
        self.operator_key = Ed25519PrivateKey.generate()
        self.operator_raw = self.operator_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw)
        self.operator_sha = sha(self.operator_raw)
        witness_raw = self.witness_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw)
        self.witness_sha = sha(witness_raw)
        self.fixture = SQLiteCASFixture.bootstrap(
            self.path, self.witness_key, self.operator_sha)
        self.scorer_sha = "2" * 64
        self.genesis = "0" * 64

    def request(self, seq=1, previous=None, execution="a" * 64,
                nonce=None):
        prev = self.fixture.highwater()[1] if previous is None else previous
        signed = {
            "schema": "exp053-operator-run-authorization-v1",
            "runner_execution_id_sha256": execution,
            "execution_attestation_sha256": "b" * 64,
            "packet_sha256": "c" * 64,
            "score_archive_sha256": "d" * 64,
            "operator_custody_reference_sha256": "e" * 64,
            "sequence": seq,
            "previous_claim_sha256": prev,
            "operator_public_key_b64": base64.b64encode(self.operator_raw).decode(),
        }
        signed["signature_hex"] = self.operator_key.sign(canonical(signed)).hex()
        raw = json.dumps(signed, sort_keys=True).encode() + b"\n"
        request = {
            "schema": witness.REQUEST_SCHEMA,
            "nonce_hex": nonce if nonce is not None else os.urandom(32).hex(),
            "sequence": seq,
            "expected_previous_sequence": seq-1,
            "previous_claim_sha256": prev,
            "claim_sha256": sha(raw),
            "runner_execution_id_sha256": execution,
            "operator_authorization_b64": base64.b64encode(raw).decode(),
        }
        return request, raw

    def checked_commit(self, seq=1, previous=None, execution="a" * 64):
        req, raw = self.request(seq, previous=previous, execution=execution)
        with patch.object(
            witness, "APPROVED_EXTERNAL_WITNESS_PUBLIC_KEY_SHA256",
            self.witness_sha), patch.object(
            witness, "APPROVED_EXTERNAL_WITNESS_ADAPTER_SHA256",
            "f" * 64), patch.object(
            witness, "APPROVED_WITNESS_OPERATOR_PUBLIC_KEY_SHA256",
            self.operator_sha), patch.object(
            witness, "_external_cas", self.fixture.cas):
            return witness.commit_independent_highwater(
                sequence=req["sequence"],
                previous_claim_sha256=req["previous_claim_sha256"],
                claim_sha256=req["claim_sha256"],
                execution_id_sha256=req["runner_execution_id_sha256"],
                operator_public_key_sha256=self.operator_sha,
                scorer_public_key_sha256=self.scorer_sha,
                operator_authorization_bytes=raw)

    def test_signed_request_verifies_before_any_db_mutation(self):
        req, _ = self.request()
        proof = verify_authenticated_request(req, self.operator_sha)
        self.assertEqual(proof["claim_sha256"], req["claim_sha256"])
        self.assertEqual(self.fixture.highwater(), (0, self.genesis, 0))
        response = self.fixture.cas(req)
        self.assertEqual(response["status"], "committed")
        self.assertEqual(self.fixture.highwater(), (1, req["claim_sha256"], 1))

    def test_genesis_persists_across_connections_and_process(self):
        self.assertEqual(self.fixture.highwater(), (0, self.genesis, 0))
        result = self.checked_commit()
        self.assertEqual(result["external_highwater_sequence"], 1)
        after = self.fixture.highwater()
        self.assertEqual(after[0], 1)
        self.assertEqual(after[2], 1)
        again = SQLiteCASFixture(self.path, self.witness_key, self.operator_sha)
        self.assertEqual(again.highwater(), after)
        program = ("import sqlite3,sys; d=sqlite3.connect(sys.argv[1]); "
                   "print(d.execute('SELECT sequence FROM witness_state').fetchone()[0])")
        self.assertEqual(subprocess.check_output(
            [sys.executable, "-c", program, str(self.path)], text=True).strip(), "1")

    def test_chained_distinct_runs_and_replayed_id(self):
        self.checked_commit()
        first = self.fixture.highwater()
        self.checked_commit(seq=2, previous=first[1], execution="b" * 64)
        self.assertEqual(self.fixture.highwater()[0], 2)
        with self.assertRaisesRegex(ValueError, "HOLD"):
            self.checked_commit(seq=3, execution="a" * 64)
        self.assertEqual(self.fixture.highwater()[0], 2)

    def test_missing_invalid_or_untrusted_authorization_denied(self):
        for kind in ("missing", "wrong_key", "wrong_claim", "bad_signature",
                     "swapped_execution", "wrong_sequence", "duplicate_field",
                     "invalid_b64"):
            with self.subTest(kind=kind):
                req, _ = self.request()
                if kind == "missing":
                    req.pop("operator_authorization_b64")
                elif kind == "wrong_key":
                    other = Ed25519PrivateKey.generate()
                    operator = other.public_key().public_bytes(
                        encoding=serialization.Encoding.Raw,
                        format=serialization.PublicFormat.Raw)
                    auth = json.loads(base64.b64decode(req["operator_authorization_b64"]))
                    auth["operator_public_key_b64"] = base64.b64encode(operator).decode()
                    auth["signature_hex"] = other.sign(canonical({
                        k: v for k, v in auth.items() if k != "signature_hex"})).hex()
                    raw = json.dumps(auth, sort_keys=True).encode()
                    req["operator_authorization_b64"] = base64.b64encode(raw).decode()
                    req["claim_sha256"] = sha(raw)
                elif kind == "wrong_claim":
                    req["claim_sha256"] = "1" * 64
                elif kind == "bad_signature":
                    auth = json.loads(base64.b64decode(req["operator_authorization_b64"]))
                    auth["signature_hex"] = "00" * 64
                    raw = json.dumps(auth).encode()
                    req["operator_authorization_b64"] = base64.b64encode(raw).decode()
                    req["claim_sha256"] = sha(raw)
                elif kind == "swapped_execution":
                    req["runner_execution_id_sha256"] = "f" * 64
                elif kind == "wrong_sequence":
                    req["sequence"] = 2
                    req["expected_previous_sequence"] = 1
                elif kind == "duplicate_field":
                    raw = base64.b64decode(req["operator_authorization_b64"])
                    raw = raw.rstrip()[:-1] + b',"schema":"exp053-operator-run-authorization-v1"}'
                    req["operator_authorization_b64"] = base64.b64encode(raw).decode()
                    req["claim_sha256"] = sha(raw)
                else:
                    req["operator_authorization_b64"] = "bad$$"
                with self.assertRaises(ValueError):
                    self.fixture.cas(req)
                self.assertEqual(self.fixture.highwater(), (0, self.genesis, 0))

    def test_duplicate_nonce_and_reused_claim_denied(self):
        one, _ = self.request(nonce="a" * 64)
        self.fixture.cas(one)
        two, _ = self.request(seq=2, execution="b" * 64, nonce="a" * 64)
        with self.assertRaises(sqlite3.IntegrityError):
            self.fixture.cas(two)
        # Reuse of the original signed claim, while changing outer sequence,
        # is rejected by signed-authorization binding before SQLite CAS.
        reused = dict(one, sequence=2, expected_previous_sequence=1,
                      nonce_hex="b" * 64, previous_claim_sha256=one["claim_sha256"])
        with self.assertRaises(ValueError):
            self.fixture.cas(reused)
        self.assertEqual(self.fixture.highwater()[0], 1)

    def test_before_commit_rolls_back_and_after_commit_is_consumed(self):
        req, _ = self.request()
        self.fixture.fail_before_commit = True
        with self.assertRaises(RuntimeError):
            self.fixture.cas(req)
        self.assertEqual(self.fixture.highwater(), (0, self.genesis, 0))
        self.fixture.fail_before_commit = False
        self.fixture.fail_after_commit = True
        with self.assertRaises(RuntimeError):
            self.fixture.cas(req)
        self.assertEqual(self.fixture.highwater(), (1, req["claim_sha256"], 1))
        self.fixture.fail_after_commit = False
        with self.assertRaises(ValueError):
            self.fixture.cas(req)

    def test_concurrent_connections_one_claim_wins(self):
        req1, _ = self.request(nonce="1" * 64)
        req2, _ = self.request(nonce="2" * 64)
        fixtures = [SQLiteCASFixture(self.path, self.witness_key,
                                     self.operator_sha) for _ in range(2)]
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(
                lambda x: try_cas(x[0].cas, x[1]), zip(fixtures, (req1, req2))))
        self.assertEqual(results.count(True), 1)
        self.assertEqual(results.count(False), 1)
        self.assertEqual(self.fixture.highwater()[0], 1)

    @unittest.skipUnless("spawn" in multiprocessing.get_all_start_methods(),
                         "spawn is unavailable")
    def test_two_spawn_processes_compete_for_same_sequence(self):
        req1, _ = self.request(nonce="a" * 64)
        req2, _ = self.request(nonce="b" * 64)
        wseed = self.witness_key.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption())
        context = multiprocessing.get_context("spawn")
        q = context.Queue()
        procs = [context.Process(
            target=process_try_cas,
            args=(str(self.path), wseed, self.operator_sha, req, q))
            for req in (req1, req2)]
        for p in procs:
            p.start()
        for p in procs:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
                p.join()
                self.fail("synthetic witness subprocess deadlocked")
            self.assertEqual(p.exitcode, 0)
        outcomes = [q.get(timeout=2) for _ in procs]
        self.assertEqual(outcomes.count(True), 1)
        self.assertEqual(outcomes.count(False), 1)
        self.assertEqual(self.fixture.highwater()[0], 1)

    def test_unpinned_or_incorrect_independent_operator_root_denied(self):
        req, _ = self.request()
        for root in ("", "f" * 64, self.witness_sha, self.scorer_sha):
            with self.subTest(root=root[:8]):
                with self.assertRaisesRegex(ValueError, "HOLD"):
                    verify_authenticated_request(req, root)
        self.assertEqual(self.fixture.highwater(), (0, self.genesis, 0))

    def test_outer_boolean_or_malformed_counter_denied_before_sqlite(self):
        for field, value in (("sequence", True),
                             ("expected_previous_sequence", True),
                             ("sequence", 1.0),
                             ("expected_previous_sequence", -1),
                             ("nonce_hex", "not-a-random-nonce")):
            req, _ = self.request()
            req[field] = value
            with self.subTest(field=field, value=str(value)):
                with self.assertRaises(ValueError):
                    self.fixture.cas(req)
                self.assertEqual(self.fixture.highwater(), (0, self.genesis, 0))

    def test_signed_private_passage_field_refused_even_with_valid_signature(self):
        req, _ = self.request()
        signed = json.loads(base64.b64decode(req["operator_authorization_b64"]))
        signed["licensed_source_article"] = "synthetic text must never be sent"
        signed["signature_hex"] = self.operator_key.sign(canonical({
            k: v for k, v in signed.items() if k != "signature_hex"})).hex()
        raw = json.dumps(signed, sort_keys=True).encode()
        req["operator_authorization_b64"] = base64.b64encode(raw).decode()
        req["claim_sha256"] = sha(raw)
        with self.assertRaisesRegex(ValueError, "HOLD"):
            self.fixture.cas(req)
        self.assertEqual(self.fixture.highwater(), (0, self.genesis, 0))

    def test_duplicate_json_and_nonfinite_operator_payload_refused(self):
        req, _ = self.request()
        raw = base64.b64decode(req["operator_authorization_b64"])
        for malformed in (
            raw.rstrip()[:-1] + b',"schema":"exp053-operator-run-authorization-v1"}',
            raw.rstrip()[:-1] + b',"bad":NaN}',
            raw.rstrip()[:-1] + b',"bad":Infinity}',
        ):
            candidate = dict(req)
            candidate["operator_authorization_b64"] = base64.b64encode(malformed).decode()
            candidate["claim_sha256"] = sha(malformed)
            with self.assertRaisesRegex(ValueError, "HOLD"):
                self.fixture.cas(candidate)
            self.assertEqual(self.fixture.highwater(), (0, self.genesis, 0))

    def test_oversized_operator_authorization_fails_before_mutation(self):
        req, _ = self.request()
        payload = b"z" * 17000
        req["operator_authorization_b64"] = base64.b64encode(payload).decode()
        req["claim_sha256"] = sha(payload)
        with self.assertRaisesRegex(ValueError, "HOLD"):
            self.fixture.cas(req)
        self.assertEqual(self.fixture.highwater(), (0, self.genesis, 0))

    def test_client_authenticates_before_reaching_external_adapter(self):
        req, _ = self.request()
        attempts = []
        with patch.object(
            witness, "APPROVED_EXTERNAL_WITNESS_PUBLIC_KEY_SHA256",
            self.witness_sha), patch.object(
            witness, "APPROVED_EXTERNAL_WITNESS_ADAPTER_SHA256", "f" * 64), patch.object(
            witness, "APPROVED_WITNESS_OPERATOR_PUBLIC_KEY_SHA256",
            self.operator_sha), patch.object(
            witness, "_external_cas", lambda r: attempts.append(r)):
            with self.assertRaisesRegex(ValueError, "HOLD"):
                witness.commit_independent_highwater(
                    sequence=req["sequence"],
                    previous_claim_sha256=req["previous_claim_sha256"],
                    claim_sha256=req["claim_sha256"],
                    execution_id_sha256=req["runner_execution_id_sha256"],
                    operator_public_key_sha256=self.operator_sha,
                    scorer_public_key_sha256=self.scorer_sha,
                    operator_authorization_bytes=b"wrong unsigned bytes")
        self.assertEqual(attempts, [])
        self.assertEqual(self.fixture.highwater(), (0, self.genesis, 0))

    def test_all_real_pins_disabled_and_no_production_adapter(self):
        self.assertEqual(witness.APPROVED_EXTERNAL_WITNESS_ADAPTER_SHA256, "")
        self.assertEqual(witness.APPROVED_WITNESS_OPERATOR_PUBLIC_KEY_SHA256, "")
        with self.assertRaisesRegex(ValueError, "HOLD"):
            witness.commit_independent_highwater(
                sequence=1, previous_claim_sha256="0" * 64,
                claim_sha256="1" * 64, execution_id_sha256="a" * 64,
                operator_public_key_sha256=self.operator_sha,
                scorer_public_key_sha256=self.scorer_sha,
                operator_authorization_bytes=b"not approved")


def try_cas(fn, request):
    try:
        fn(request)
        return True
    except (ValueError, sqlite3.IntegrityError):
        return False


def process_try_cas(path, ephemeral_witness_seed, operator_sha, request, q):
    key = Ed25519PrivateKey.from_private_bytes(ephemeral_witness_seed)
    q.put(try_cas(SQLiteCASFixture(path, key, operator_sha).cas, request))


if __name__ == "__main__":
    unittest.main()
