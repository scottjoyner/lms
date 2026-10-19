"""Synthetic-only Ed25519 prime-subgroup validation of human roster public keys."""

import base64
import copy
import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
import exp053_reviewer_roster as roster

# Published RFC 8032 Ed25519 verification public keys, NOT real reviewers.
RFC8032 = [
    bytes.fromhex("d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a"),
    bytes.fromhex("3d4017c3e843895a92b70aa74d1b7ebc9c982ccf2ec4968cc0cd55f12af4660c"),
    bytes.fromhex("fc51cd8e6218a1a38da47ed00230f0580816ed13ba3303ac5deb911548908025"),
]


class TestWeakReviewerKeys(unittest.TestCase):
    def setUp(self):
        t = tempfile.TemporaryDirectory(prefix="exp053-weak-public-key-synthetic-")
        self.addCleanup(t.cleanup)
        self.home = Path(t.name)
        self.path = self.home / "SYNTHETIC-ROSTER-ONLY.json"
        self.content = {"schema": roster.SCHEMA, "packet_sha256": "c" * 64,
                        "owner_custody_reference_sha256": "d" * 64,
                        "blind_assignments_confirmed": True,
                        "slots": {
                            "A": {"reviewer_id": "synthetic_A", "role": "independent",
                                  "public_key_b64": base64.b64encode(RFC8032[0]).decode()},
                            "B": {"reviewer_id": "synthetic_B", "role": "independent",
                                  "public_key_b64": base64.b64encode(RFC8032[1]).decode()},
                        }}

    def load(self, public_key_a=None):
        payload = copy.deepcopy(self.content)
        if public_key_a is not None:
            payload["slots"]["A"]["public_key_b64"] = base64.b64encode(public_key_a).decode()
        blob = json.dumps(payload, sort_keys=True).encode()
        self.path.write_bytes(blob)
        os.chmod(self.path, 0o600)
        return roster.load_roster(self.path, "c" * 64, hashlib.sha256(blob).hexdigest(), ROOT)

    def test_published_rfc_ed25519_prime_group_vectors_accepted(self):
        for p in RFC8032:
            with self.subTest(public_key=p.hex()[:12]):
                self.assertTrue(roster._prime_order_ed25519_public_key(p))
        self.assertEqual(set(self.load()["keys"]), {"synthetic_A", "synthetic_B"})
        self.assertEqual(set(self.load(RFC8032[2])["keys"]), {"synthetic_A", "synthetic_B"})

    def test_weak_keys_refused_with_exact_matching_owner_hash(self):
        field = roster._FIELD
        identity = b"\x01" + b"\x00" * 31
        weak = (
            identity, b"\x00" * 32, b"\xff" * 32,
            field.to_bytes(32, "little"),  # Noncanonical y field.
            identity[:-1] + b"\x80",  # x=0, invalid sign bit.
            (field - 1).to_bytes(32, "little"),  # Order-2 point.
        )
        for p in weak:
            with self.subTest(key=p.hex()[:16]):
                self.assertFalse(roster._prime_order_ed25519_public_key(p))
                with self.assertRaisesRegex(ValueError, "^HOLD EXP053 ROSTER: invalid or low-order"):
                    self.load(p)

    def test_mixed_torsion_public_key_is_denied(self):
        # Add a nontrivial 2-torsion point to a legitimate RFC public key.
        # The resulting point is not in the prime-order subgroup.
        raw = RFC8032[0]
        packed = int.from_bytes(raw, "little")
        sign = packed >> 255
        p = roster._FIELD
        y = packed & ((1 << 255) - 1)
        x2 = ((y*y-1) * pow((roster._D*y*y+1) % p, p-2, p)) % p
        x = pow(x2, (p+3)//8, p)
        if x*x % p != x2:
            x = x*roster._SQRT_MINUS_ONE % p
        if x & 1 != sign:
            x = p-x
        good = (x, y, 1, x*y % p)
        order2 = (0, p-1, 1, 0)
        mixed = roster._edwards_add(good, order2)
        mx, my, mz, _ = mixed
        mz_inv = pow(mz, p-2, p)
        x_affine, y_affine = mx*mz_inv % p, my*mz_inv % p
        raw_mixed = (y_affine | ((x_affine & 1) << 255)).to_bytes(32, "little")
        self.assertFalse(roster._prime_order_ed25519_public_key(raw_mixed))
        with self.assertRaisesRegex(ValueError, "HOLD EXP053 ROSTER"):
            self.load(raw_mixed)

    def test_direct_review_verifier_rejects_identity_point_signature(self):
        from exp053_race_review import verify_signed
        identity = b"\x01" + b"\x00" * 31
        fake = {"schema": "exp053-review-v1", "reviewer_id": "synthetic_A",
                "role": "independent", "packet_sha256": "synthetic-only",
                "attestation": ("I independently reviewed these source passages and responses; "
                                "no model predictions or other reviewer labels were provided to me."),
                "decisions": []}
        fake["signature"] = (bytes.fromhex("58" + "66" * 31) +
                             b"\x01" + b"\x00" * 31).hex()
        trust = {"synthetic_A": base64.b64encode(identity).decode(),
                 "synthetic_B": base64.b64encode(RFC8032[1]).decode()}
        # This record previously passed verify_signed despite having no
        # independent private-key holder. No actual human reviewed anything.
        with self.assertRaisesRegex(ValueError, "HOLD EXP053"):
            verify_signed(fake, trust)

    def test_synthetic_fresh_cryptography_public_keys_accepted(self):
        try:
            from cryptography.hazmat.primitives import serialization
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        except ImportError:
            self.skipTest("cryptography unavailable")
        for _ in range(8):
            private = Ed25519PrivateKey.generate()  # Disposable unit-test key.
            public = private.public_key().public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw)
            self.assertTrue(roster._prime_order_ed25519_public_key(public))
            self.assertEqual(set(self.load(public)["keys"]), {"synthetic_A", "synthetic_B"})

    def test_underlying_provider_signature_does_not_establish_key_possession(self):
        try:
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        except ImportError:
            self.skipTest("cryptography unavailable")
        identity = b"\x01" + b"\x00" * 31
        public = Ed25519PublicKey.from_public_bytes(identity)
        # Purely synthetic, publicly known Ed25519 base point and scalar 1.
        sample = bytes.fromhex("58" + "66" * 31) + b"\x01" + b"\x00" * 31
        for msg in (b"synthetic nonreview message one", b"synthetic nonreview message two"):
            public.verify(sample, msg)  # Valid on the tested OpenSSL runtime.
        with self.assertRaisesRegex(ValueError, "HOLD EXP053 ROSTER"):
            self.load(identity)


if __name__ == "__main__":
    unittest.main()
