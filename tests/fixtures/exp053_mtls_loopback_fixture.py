"""Disposable one-request mutual-TLS witness; never import from research.

All private certs, private signing keys, and SQLite state are synthetic test
fixtures, generated in tempfile dirs. No production or internet listener.
"""
import hashlib
import ipaddress
import json
import os
import socket
import ssl
import struct
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

from exp053_execution_attestation import unambiguous_json

MAX_FRAME = 32 * 1024


def _read_exact(stream, n):
    data = bytearray()
    while len(data) < n:
        part = stream.recv(n - len(data))
        if not part:
            raise ConnectionError("synthetic TLS witness closed")
        data.extend(part)
    return bytes(data)


def _recv_frame(stream):
    size = struct.unpack("!I", _read_exact(stream, 4))[0]
    if not 1 <= size <= MAX_FRAME:
        raise ValueError("synthetic TLS witness: oversized/empty request")
    return _read_exact(stream, size)


def _send_frame(stream, payload):
    if not 1 <= len(payload) <= MAX_FRAME:
        raise ValueError("synthetic TLS witness: oversized/empty response")
    stream.sendall(struct.pack("!I", len(payload)) + payload)


class SyntheticCA:
    """Ephemeral test-only X509 CA, TLS server and two TLS clients."""

    def __init__(self, directory):
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import ec
        from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        now = datetime.now(timezone.utc)
        self.key = ec.generate_private_key(ec.SECP256R1())
        name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "EXP053 test-only CA")])
        self.cert = (
            x509.CertificateBuilder().subject_name(name).issuer_name(name)
            .public_key(self.key.public_key()).serial_number(x509.random_serial_number())
            .not_valid_before(now-timedelta(minutes=1))
            .not_valid_after(now+timedelta(hours=2))
            .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
            .add_extension(x509.KeyUsage(
                digital_signature=True, content_commitment=False, key_encipherment=False,
                data_encipherment=False, key_agreement=False, key_cert_sign=True,
                crl_sign=True, encipher_only=False, decipher_only=False), critical=True)
            .sign(self.key, hashes.SHA256())
        )
        self.ca_pem = self.directory / "synthetic-ca.pem"
        self.ca_pem.write_bytes(self.cert.public_bytes(serialization.Encoding.PEM))
        self.ExtendedKeyUsageOID = ExtendedKeyUsageOID
        self._x509 = x509
        self._hashes = hashes
        self._serialization = serialization
        self._ec = ec

    def issue(self, name, *, client):
        x509 = self._x509
        key = self._ec.generate_private_key(self._ec.SECP256R1())
        now = datetime.now(timezone.utc)
        subject = x509.Name([x509.NameAttribute(
            x509.oid.NameOID.COMMON_NAME, "EXP053 synthetic " + name)])
        eku = (self.ExtendedKeyUsageOID.CLIENT_AUTH if client
               else self.ExtendedKeyUsageOID.SERVER_AUTH)
        builder = (
            x509.CertificateBuilder().subject_name(subject)
            .issuer_name(self.cert.subject).public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now-timedelta(minutes=1))
            .not_valid_after(now+timedelta(hours=1))
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .add_extension(x509.ExtendedKeyUsage([eku]), critical=True)
            .add_extension(x509.KeyUsage(
                digital_signature=True, content_commitment=False,
                key_encipherment=False, data_encipherment=False, key_agreement=False,
                key_cert_sign=False, crl_sign=False, encipher_only=False,
                decipher_only=False), critical=True)
        )
        if not client:
            builder = builder.add_extension(
                x509.SubjectAlternativeName([x509.IPAddress(
                    ipaddress.ip_address("127.0.0.1"))]), critical=False)
        cert = builder.sign(self.key, self._hashes.SHA256())
        cert_path = self.directory / (name + ".cert.pem")
        key_path = self.directory / (name + ".key.pem")
        cert_path.write_bytes(cert.public_bytes(self._serialization.Encoding.PEM))
        key_path.write_bytes(key.private_bytes(
            encoding=self._serialization.Encoding.PEM,
            format=self._serialization.PrivateFormat.PKCS8,
            encryption_algorithm=self._serialization.NoEncryption()))
        os.chmod(key_path, 0o600)
        return (cert_path, key_path,
                hashlib.sha256(cert.public_bytes(self._serialization.Encoding.DER)).hexdigest())


class LoopbackMTLSWitness:
    """Single-use loopback TLS server around a synthetic SQLite witness CAS."""

    def __init__(self, directory, sqlite_fixture):
        self.root = Path(directory)
        self.ca = SyntheticCA(self.root / "pki")
        self.server_cert = self.ca.issue("server", client=False)
        self.good_client = self.ca.issue("authorized-client", client=True)
        self.other_client = self.ca.issue("wrong-client", client=True)
        self.untrusted_ca = SyntheticCA(self.root / "untrusted-pki")
        self.untrusted_client = self.untrusted_ca.issue("untrusted-client", client=True)
        self.sqlite = sqlite_fixture

    def _server_context(self):
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.minimum_version = ssl.TLSVersion.TLSv1_3
        ctx.maximum_version = ssl.TLSVersion.TLSv1_3
        ctx.verify_mode = ssl.CERT_REQUIRED
        ctx.load_cert_chain(str(self.server_cert[0]), str(self.server_cert[1]))
        ctx.load_verify_locations(cafile=str(self.ca.ca_pem))
        return ctx

    def _client_context(self, cert, *, trusted_ca=None):
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ctx.minimum_version = ssl.TLSVersion.TLSv1_3
        ctx.maximum_version = ssl.TLSVersion.TLSv1_3
        ctx.verify_mode = ssl.CERT_REQUIRED
        ctx.check_hostname = True
        ctx.load_verify_locations(cafile=str(
            self.ca.ca_pem if trusted_ca is None else trusted_ca))
        if cert is not None:
            ctx.load_cert_chain(str(cert[0]), str(cert[1]))
        return ctx

    def exchange(self, payload, *, client="good", hostname="127.0.0.1",
                 client_trust_ca=None):
        """Start exactly one local TLS connection, then join and close server."""
        cert = {"good": self.good_client, "other": self.other_client,
                "untrusted": self.untrusted_client, "none": None}[client]
        if not isinstance(payload, bytes):
            payload = json.dumps(payload, sort_keys=True).encode()
        errors = []
        ready = threading.Event()
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            listener.settimeout(3)
            port = listener.getsockname()[1]
            server_context = self._server_context()

            def serve():
                ready.set()
                try:
                    conn, _ = listener.accept()
                    with conn:
                        conn.settimeout(3)
                        with server_context.wrap_socket(conn, server_side=True) as tls:
                            if tls.version() != "TLSv1.3":
                                raise ssl.SSLError("TLSv1.3 required")
                            peer = tls.getpeercert(binary_form=True)
                            if (not peer or hashlib.sha256(peer).hexdigest() !=
                                    self.good_client[2]):
                                raise PermissionError("unapproved client leaf")
                            request = unambiguous_json(_recv_frame(tls))
                            response = self.sqlite.cas(request)
                            _send_frame(tls, json.dumps(
                                response, sort_keys=True).encode())
                except (OSError, ValueError, PermissionError, ConnectionError) as e:
                    errors.append(type(e).__name__)

            thread = threading.Thread(target=serve, daemon=True)
            thread.start()
            ready.wait(2)
            try:
                client_ctx = self._client_context(cert, trusted_ca=client_trust_ca)
                with socket.create_connection(("127.0.0.1", port), timeout=3) as raw:
                    raw.settimeout(3)
                    with client_ctx.wrap_socket(raw, server_hostname=hostname) as tls:
                        if tls.version() != "TLSv1.3":
                            raise ssl.SSLError("TLSv1.3 required")
                        _send_frame(tls, payload)
                        return unambiguous_json(_recv_frame(tls))
            finally:
                thread.join(timeout=4)
                if thread.is_alive():
                    raise RuntimeError("synthetic TLS witness thread did not stop")
