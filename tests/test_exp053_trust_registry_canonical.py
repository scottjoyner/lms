"""Synthetic direct signer-verifier trust-map and receipt text admission tests."""

import base64
import copy
import sys
import unittest
from unittest.mock import patch
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'research'))
import exp053_race_review as verify

ATTEST=('I independently reviewed these source passages and responses; '
        'no model predictions or other reviewer labels were provided to me.')
PUB_A=bytes.fromhex('d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a')
PUB_B=bytes.fromhex('3d4017c3e843895a92b70aa74d1b7ebc9c982ccf2ec4968cc0cd55f12af4660c')
PUB_C=bytes.fromhex('fc51cd8e6218a1a38da47ed00230f0580816ed13ba3303ac5deb911548908025')
ALPHABET='ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/'


def alternate_base64(raw):
    """Change unused last-quantum bits without changing the decoded key."""
    canonical=base64.b64encode(raw).decode()
    assert canonical.endswith('=') and not canonical.endswith('==')
    idx=ALPHABET.index(canonical[-2])
    alias=canonical[:-2]+ALPHABET[idx|1]+'='
    assert alias!=canonical
    assert base64.b64decode(alias,validate=True)==raw
    return alias


class SyntheticDirectTrust(unittest.TestCase):
    def setUp(self):
        self.review={'schema':'exp053-review-v1','reviewer_id':'synthetic_A',
                     'role':'independent','packet_sha256':'SYNTHETIC-ONLY',
                     'attestation':ATTEST,'decisions':[],
                     'signature':'00'*64}
        self.trust={'synthetic_A':base64.b64encode(PUB_A).decode(),
                    'synthetic_B':base64.b64encode(PUB_B).decode()}

    def test_same_public_bytes_different_base64_alias_refused_before_crypto(self):
        self.trust['synthetic_B']=alternate_base64(PUB_A)
        with patch.dict(sys.modules,{'cryptography.hazmat.primitives.asymmetric.ed25519':None}):
            with self.assertRaisesRegex(ValueError,'HOLD EXP053: trust registry'):
                verify.verify_signed(self.review,self.trust)

    def test_invalid_secondary_identity_key_refused_before_crypto(self):
        self.trust['synthetic_B']=base64.b64encode(b'\x01'+b'\x00'*31).decode()
        with patch.dict(sys.modules,{'cryptography.hazmat.primitives.asymmetric.ed25519':None}):
            with self.assertRaisesRegex(ValueError,'HOLD EXP053: trust registry'):
                verify.verify_signed(self.review,self.trust)

    def test_unhashable_nonstrings_and_invalid_secondary_refused_safely(self):
        for bad in ([],{},None,7,True,b'nonstring','', '!!!'):
            with self.subTest(bad=repr(bad)):
                self.trust['synthetic_B']=bad
                with self.assertRaisesRegex(ValueError,'HOLD EXP053'):
                    verify.verify_signed(self.review,self.trust)

    def test_trust_map_size_is_bounded_to_owner_roster_slots(self):
        for n in (0, 1, 4, 25):
            with self.subTest(count=n):
                mapping={f"synthetic_{i}":base64.b64encode(PUB_A).decode()
                         for i in range(n)}
                with self.assertRaisesRegex(ValueError,'HOLD EXP053'):
                    verify.verify_signed(self.review,mapping)

    def test_trust_map_invalid_identity_names_are_refused(self):
        for bad in ('',7,None):
            with self.subTest(bad=repr(bad)):
                candidate={bad:self.trust['synthetic_A'],
                           'synthetic_A':self.trust['synthetic_A'],
                           'synthetic_B':self.trust['synthetic_B']}
                with self.assertRaisesRegex(ValueError,'HOLD EXP053'):
                    verify.verify_signed(self.review,candidate)

    def test_noncanonical_encoding_is_refused_even_for_current_signer(self):
        self.trust['synthetic_A']=alternate_base64(PUB_A)
        with self.assertRaisesRegex(ValueError,'HOLD EXP053: trust registry'):
            verify.verify_signed(self.review,self.trust)

    def test_canonical_signature_lowercase_hex_required(self):
        variants=['00 '*63+'00','A0'*64,'0'*127,'0'*129,None,[],123,'0x'+'0'*126]
        for sig in variants:
            with self.subTest(sig=repr(sig)[:30]):
                record={**self.review,'signature':sig}
                with patch.dict(sys.modules,{'cryptography.hazmat.primitives.asymmetric.ed25519':None}):
                    with self.assertRaisesRegex(ValueError,
                                                'HOLD EXP053: signed reviewer signature must'):
                        verify.verify_signed(record,self.trust)

    def test_valid_trust_without_crypto_backend_still_hold(self):
        self.trust['synthetic_C']=base64.b64encode(PUB_C).decode()
        with patch.dict(sys.modules,{'cryptography.hazmat.primitives.asymmetric.ed25519':None}):
            with self.assertRaisesRegex(ValueError,'HOLD EXP053'):
                verify.verify_signed(self.review,self.trust)

    def test_valid_distinct_ephemeral_signers_success_when_backend_present(self):
        try:
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
            from cryptography.hazmat.primitives import serialization
        except ImportError:
            self.skipTest('optional cryptography absent in hosted research CI')
        a,b=Ed25519PrivateKey.generate(),Ed25519PrivateKey.generate()
        def pub(k):
            return base64.b64encode(k.public_key().public_bytes(
                serialization.Encoding.Raw,serialization.PublicFormat.Raw)).decode()
        self.trust={'synthetic_A':pub(a),'synthetic_B':pub(b)}
        record={k:v for k,v in self.review.items() if k!='signature'}
        self.review['signature']=a.sign(verify.canonical(record)).hex()
        self.assertEqual(verify.verify_signed(self.review,self.trust),'synthetic_A')
        uppercase=copy.deepcopy(self.review)
        uppercase['signature']=self.review['signature'].upper()
        if uppercase['signature']!=self.review['signature']:
            with self.assertRaisesRegex(ValueError,'HOLD EXP053'):
                verify.verify_signed(uppercase,self.trust)


if __name__=='__main__':
    unittest.main()
