"""Fictional log-probabilities only: no licensed RACE or actual model data."""

import hashlib
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
import exp053_execution_attestation as scorer


class NumericJsonFidelity(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="exp053-invented-json-score-")
        self.addCleanup(temp.cleanup)
        self.private = Path(temp.name)
        self.file = self.private / "invented-not-scientific-scores.json"

    def parsed(self, source):
        return scorer.unambiguous_json(source.encode())

    def pinned_scores(self, raw):
        payload = raw.encode()
        self.file.write_bytes(payload)
        os.chmod(self.file, 0o600)
        return scorer.read_attested_scores(
            self.file, self.private, hashlib.sha256(payload).hexdigest())

    def test_nonzero_decimal_that_rounds_to_zero_is_denied(self):
        numbers = ("-1e-999", "1e-999", "1E-99999", "-2.0E-320000",
                   "-0.000000001e-999", "9.9e-99999", "-1e-324")
        for value in numbers:
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "HOLD"):
                self.parsed('{"synthetic_logp":' + value + '}')

    def test_float_overflow_is_denied(self):
        for value in ("1e999", "-1e999", "1.8e308", "-9.9e9999"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "HOLD"):
                self.parsed('{"synthetic_logp":' + value + '}')

    def test_representable_and_genuine_zero_finite_values_preserved(self):
        values = ("-2.5", "0.0", "-0.0", "-0.000e-99999", "0E+99999",
                  "2.2250738585072014e-308", "5e-324", "-5e-324",
                  "1.0e-7", "1.7976931348623157e308")
        for value in values:
            with self.subTest(value=value):
                got = self.parsed('{"synthetic_logp":' + value + '}')["synthetic_logp"]
                self.assertTrue(math.isfinite(got))
                self.assertEqual(got, float(value))
                if float(value) != 0.0:
                    self.assertNotEqual(got, 0.0)

    def test_nested_float_overflow_underflow_denied(self):
        for raw in ('{"invented":{"logprob":-1e-999}}',
                    '{"invented":[1.0,1e999]}',
                    '[{"case":"invented","gold_logp":-1e-999}]'):
            with self.subTest(raw=raw), self.assertRaisesRegex(ValueError,"HOLD"):
                self.parsed(raw)

    def test_unambiguous_json_retains_valid_signed_claim_integer_fields(self):
        got = self.parsed('{"synthetic_sequence":1,"synthetic_counter":0}')
        self.assertEqual(got, {"synthetic_sequence":1,"synthetic_counter":0})
        self.assertIs(type(got["synthetic_sequence"]), int)

    def test_pinned_score_archive_rejects_underflow_even_if_bytes_match(self):
        with self.assertRaisesRegex(ValueError,"HOLD"):
            self.pinned_scores('[{"synthetic_case":"fiction","gold_logp":-1e-999}]')

    def test_pinned_score_archive_rejects_overflow_even_if_bytes_match(self):
        with self.assertRaisesRegex(ValueError,"HOLD"):
            self.pinned_scores('[{"synthetic_case":"fiction","gold_logp":1e999}]')

    def test_pinned_score_archive_preserves_valid_finite_scores(self):
        rows = self.pinned_scores(
            '[{"synthetic_case":"fiction","gold_logp":-0.125,"foil_logp":-3e-5}]')
        self.assertEqual(rows[0]["gold_logp"], -0.125)
        self.assertEqual(rows[0]["foil_logp"], -3e-5)

    def test_pinned_score_archive_preserves_genuine_negative_zero(self):
        rows=self.pinned_scores('[{"synthetic_case":"fiction","gold_logp":-0.0}]')
        self.assertEqual(rows[0]["gold_logp"],0.0)
        self.assertEqual(math.copysign(1.0,rows[0]["gold_logp"]),-1.0)

    def test_explicit_json_nan_and_infinity_remain_denied(self):
        for num in ("NaN","Infinity","-Infinity"):
            with self.subTest(num=num), self.assertRaisesRegex(ValueError, "HOLD"):
                self.parsed('{"synthetic_logp":'+num+'}')

    def test_duplicate_json_keys_remain_denied(self):
        with self.assertRaisesRegex(ValueError,"HOLD"):
            self.parsed('{"case":"fiction","case":"other"}')

    def test_invalid_json_and_wrong_pinned_hash_still_denied(self):
        with self.assertRaisesRegex(ValueError,"HOLD"):
            self.parsed('{"oops":0.5')
        payload=b'[{"logp":-1.25}]'
        self.file.write_bytes(payload)
        self.file.chmod(0o600)
        with self.assertRaisesRegex(ValueError,"HOLD"):
            scorer.read_attested_scores(self.file,self.private,"0"*64)

    def test_text_of_fake_scores_is_not_leaked_in_parser_errors(self):
        invented='fictitious_secret_dummy_12345'
        try:
            self.parsed('{"'+invented+'":1e999}')
        except ValueError as error:
            self.assertNotIn(invented,str(error))
        else:
            self.fail('overflow must hold')


if __name__ == "__main__":
    unittest.main()
