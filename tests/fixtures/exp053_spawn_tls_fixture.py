"""Spawned loopback witness harness ONLY for disposable EXP053 software tests.

The test-only witness's synthetic key SEED is passed to its child. This
deliberately provides *process* isolation, NOT independent key custody.
"""
import hashlib
import json
import multiprocessing
import os
import socket
import ssl
from pathlib import Path

from exp053_execution_attestation import unambiguous_json
from exp053_mtls_loopback_fixture import _recv_frame, _send_frame
from exp053_sqlite_cas_fixture import SQLiteCASFixture


def _serve_in_spawn(pipe, *, database, witness_seed, operator_key_sha,
                    ca_cert, server_cert, server_key, authorized_client_sha,
                    failure=None):
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    key = Ed25519PrivateKey.from_private_bytes(witness_seed)
    fixture = SQLiteCASFixture(database, key, operator_key_sha)
    if failure == "before_commit":
        fixture.fail_before_commit = True
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.minimum_version = ssl.TLSVersion.TLSv1_3
    context.maximum_version = ssl.TLSVersion.TLSv1_3
    context.verify_mode = ssl.CERT_REQUIRED
    context.load_cert_chain(str(server_cert), str(server_key))
    context.load_verify_locations(cafile=str(ca_cert))
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            listener.settimeout(3)
            pipe.send(("ready", listener.getsockname()[1], os.getpid()))
            conn, _ = listener.accept()
            with conn:
                conn.settimeout(3)
                with context.wrap_socket(conn, server_side=True) as tls:
                    if tls.version() != "TLSv1.3":
                        raise ssl.SSLError("unexpected TLS version")
                    peer = tls.getpeercert(binary_form=True)
                    if (not peer or
                            hashlib.sha256(peer).hexdigest() !=
                            authorized_client_sha):
                        raise PermissionError("wrong pinned synthetic client identity")
                    request = unambiguous_json(_recv_frame(tls))
                    result = fixture.cas(request)
                    if failure == "after_commit":
                        os._exit(73)  # TEST PROCESS ONLY; durable DB retains claim
                    _send_frame(tls, json.dumps(result, sort_keys=True).encode())
    except (OSError, ValueError, ConnectionError, PermissionError, RuntimeError) as e:
        pipe.send(("denied", type(e).__name__))
    finally:
        pipe.close()


class SpawnedTLSWitness:
    """Exactly one TLS client session handled by a separate spawned process."""

    def __init__(self, tls_fixture, db_path, witness_key, operator_sha,
                 *, failure=None):
        from cryptography.hazmat.primitives import serialization
        self.front = tls_fixture
        self.ctx = multiprocessing.get_context("spawn")
        self.parent, child = self.ctx.Pipe()
        self.process = self.ctx.Process(
            target=_serve_in_spawn,
            kwargs={
                "pipe": child,
                "database": str(Path(db_path)),
                "witness_seed": witness_key.private_bytes(
                    encoding=serialization.Encoding.Raw,
                    format=serialization.PrivateFormat.Raw,
                    encryption_algorithm=serialization.NoEncryption()),
                "operator_key_sha": operator_sha,
                "ca_cert": str(tls_fixture.ca.ca_pem),
                "server_cert": str(tls_fixture.server_cert[0]),
                "server_key": str(tls_fixture.server_cert[1]),
                "authorized_client_sha": tls_fixture.good_client[2],
                "failure": failure,
            },
        )
        self.port = None
        self.pid = None
        self._child_pipe = child

    def start(self):
        self.process.start()
        self._child_pipe.close()
        if not self.parent.poll(8):
            self.close()
            raise TimeoutError("synthetic child TLS witness did not start")
        state = self.parent.recv()
        if state[0] != "ready" or type(state[1]) is not int:
            self.close()
            raise ValueError("synthetic TLS witness failed startup")
        self.port, self.pid = state[1], state[2]
        return self

    def exchange(self, message, *, client="good", hostname="127.0.0.1",
                 trusted_ca=None):
        if self.port is None:
            raise RuntimeError("test TLS child is not running")
        cert = {"good": self.front.good_client,
                "other": self.front.other_client,
                "untrusted": self.front.untrusted_client,
                "none": None}[client]
        context = self.front._client_context(cert, trusted_ca=trusted_ca)
        data = (message if isinstance(message, bytes) else
                json.dumps(message, sort_keys=True).encode())
        with socket.create_connection(("127.0.0.1", self.port), timeout=3) as raw:
            raw.settimeout(3)
            with context.wrap_socket(raw, server_hostname=hostname) as tls:
                if tls.version() != "TLSv1.3":
                    raise ssl.SSLError("TLSv1.3 required")
                _send_frame(tls, data)
                return unambiguous_json(_recv_frame(tls))

    def close(self):
        if self.process.pid:
            self.process.join(timeout=5)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=3)
                raise RuntimeError("disposable TLS child failed to exit")
        self.parent.close()

    def __enter__(self):
        return self.start()

    def __exit__(self, *_args):
        self.close()
