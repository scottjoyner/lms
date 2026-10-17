"""Disposable SQLite-CAS witness for EXP053 tests ONLY.

Never import from research/runtime code; never connect to a production
endpoint. Signing keys are ephemeral and synthetic.
"""
import base64
import os
import sqlite3
from pathlib import Path

from exp053_external_witness import REQUEST_FIELDS, REQUEST_SCHEMA, RESPONSE_SCHEMA
from exp053_race_review import canonical
from exp053_witness_request_auth import verify_authenticated_request


def digest64(s):
    return isinstance(s, str) and len(s) == 64 and all(
        c in "0123456789abcdef" for c in s)


class SQLiteCASFixture:
    """Test-only CAS with fresh connections and atomic SQLite transactions."""

    def __init__(self, path, signing_key, expected_operator_key_sha256):
        self.path = Path(path)
        self.signing_key = signing_key
        self.operator_key_sha256 = expected_operator_key_sha256
        self.fail_before_commit = False
        self.fail_after_commit = False

    @classmethod
    def bootstrap(cls, path, signing_key, expected_operator_key_sha256):
        path = Path(path)
        if path.exists() or not path.parent.is_dir():
            raise ValueError("test database must have absent path and existing parent")
        conn = sqlite3.connect(path, timeout=3.0, isolation_level=None)
        try:
            conn.execute("PRAGMA journal_mode=DELETE")
            conn.execute("PRAGMA synchronous=FULL")
            conn.execute("""
                CREATE TABLE witness_state (
                    id INTEGER PRIMARY KEY CHECK (id=1),
                    sequence INTEGER NOT NULL CHECK(sequence>=0),
                    claim_sha256 TEXT NOT NULL
                )""")
            conn.execute("""
                CREATE TABLE witness_claims (
                    sequence INTEGER PRIMARY KEY,
                    execution_id_sha256 TEXT NOT NULL UNIQUE,
                    nonce_hex TEXT NOT NULL UNIQUE,
                    claim_sha256 TEXT NOT NULL UNIQUE,
                    previous_claim_sha256 TEXT NOT NULL
                )""")
            conn.execute(
                "INSERT INTO witness_state(id,sequence,claim_sha256) VALUES(1,0,?)",
                ("0" * 64,))
        finally:
            conn.close()
        os.chmod(path, 0o600)
        return cls(path, signing_key, expected_operator_key_sha256)

    def connect(self):
        if not self.path.is_file():
            raise ValueError("test database missing")
        conn = sqlite3.connect(
            self.path, timeout=3.0, isolation_level=None, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=DELETE")
        conn.execute("PRAGMA synchronous=FULL")
        return conn

    def highwater(self):
        conn = self.connect()
        try:
            state = conn.execute(
                "SELECT sequence,claim_sha256 FROM witness_state WHERE id=1"
            ).fetchone()
            count = conn.execute("SELECT COUNT(*) FROM witness_claims").fetchone()[0]
            return *state, count
        finally:
            conn.close()

    def validate(self, req):
        # Independently provisioned operator root; not a client-supplied
        # fingerprint. Invalid auth must fail BEFORE SQLite BEGIN IMMEDIATE.
        verify_authenticated_request(req, self.operator_key_sha256)
        if not isinstance(req, dict) or set(req) != REQUEST_FIELDS:
            raise ValueError("synthetic CAS request has wrong fields")
        if req["schema"] != REQUEST_SCHEMA:
            raise ValueError("incorrect CAS request schema")
        if (type(req["sequence"]) is not int or req["sequence"] < 1 or
                type(req["expected_previous_sequence"]) is not int or
                req["expected_previous_sequence"] != req["sequence"] - 1):
            raise ValueError("invalid synthetic sequence")
        for name in ("nonce_hex", "previous_claim_sha256", "claim_sha256",
                     "runner_execution_id_sha256"):
            if not digest64(req[name]):
                raise ValueError("invalid SHA256/nonce field")

    def cas(self, request):
        """Atomically commit CAS+uniqueness BEFORE signing response."""
        self.validate(request)
        conn = self.connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            state = conn.execute(
                "SELECT sequence,claim_sha256 FROM witness_state WHERE id=1"
            ).fetchone()
            if state is None:
                raise ValueError("missing witness state")
            sequence, previous = state
            if (request["sequence"] != sequence + 1 or
                    request["expected_previous_sequence"] != sequence or
                    request["previous_claim_sha256"] != previous):
                raise ValueError("stale or competing CAS")
            conn.execute(
                """INSERT INTO witness_claims(
                   sequence,execution_id_sha256,nonce_hex,claim_sha256,
                   previous_claim_sha256) VALUES(?,?,?,?,?)""",
                (request["sequence"], request["runner_execution_id_sha256"],
                 request["nonce_hex"], request["claim_sha256"],
                 request["previous_claim_sha256"]))
            if self.fail_before_commit:
                raise RuntimeError("synthetic interruption before commit")
            changed = conn.execute(
                """UPDATE witness_state SET sequence=?,claim_sha256=?
                   WHERE id=1 AND sequence=? AND claim_sha256=?""",
                (request["sequence"], request["claim_sha256"],
                 sequence, previous))
            if changed.rowcount != 1:
                raise RuntimeError("witness lost update")
            conn.execute("COMMIT")
        except Exception:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()
        if self.fail_after_commit:
            raise RuntimeError("synthetic lost response AFTER committed CAS")
        from cryptography.hazmat.primitives import serialization
        public = self.signing_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw)
        response = {
            "schema": RESPONSE_SCHEMA,
            "nonce_hex": request["nonce_hex"],
            "sequence": request["sequence"],
            "previous_claim_sha256": request["previous_claim_sha256"],
            "claim_sha256": request["claim_sha256"],
            "runner_execution_id_sha256": request["runner_execution_id_sha256"],
            "witness_counter": request["sequence"],
            "status": "committed",
            "witness_public_key_b64": base64.b64encode(public).decode(),
        }
        response["signature_hex"] = self.signing_key.sign(canonical(response)).hex()
        return response
