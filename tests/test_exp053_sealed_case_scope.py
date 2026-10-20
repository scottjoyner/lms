"""Fully invented packet/answer-key structures; no RACE or human signing."""

import copy
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
import exp053_race_review as gate

MISSING = object()


class SyntheticSealedScope(unittest.TestCase):
    def setUp(self):
        self.packet = {
            "schema": "exp053-packet-v1", "seed": "invented-test-seed",
            "source_sha256": "0" * 64, "protocol_sha256": "1" * 64,
            "license": "synthetic-not-RACE", "cases": [
                {"case_id": f"invented-{i:03}",
                 "article": "PUBLIC-SAFE FAKE STRING: no real article",
                 "question": "Invented question only?",
                 "options": {"A": "fake", "B": "dummy", "C": "fiction", "D": "test"}}
                for i in range(64)]}
        self.answers = {f"invented-{i:03}": "ABCD"[i % 4]
                        for i in range(64)}
        self.seal = {"packet_sha256": gate.digest(self.packet),
                     "answer_key": self.answers}
        self.manifest = {"sealed_answer_key_sha256": gate.digest(self.answers),
                         "packet_sha256": gate.digest(self.packet)}

    def validate(self, packet=MISSING, seal=MISSING, manifest=MISSING):
        return gate._validate_sealed_case_scope(
            self.packet if packet is MISSING else packet,
            self.seal if seal is MISSING else seal,
            self.manifest if manifest is MISSING else manifest)

    def test_complete_64_case_synthetic_key_set_accepted_by_structure_only(self):
        self.assertEqual(self.validate(), self.answers)

    def test_40_of_64_is_not_complete_even_with_self_consistent_digest(self):
        partial = dict(list(self.answers.items())[:40])
        seal = {"packet_sha256": gate.digest(self.packet), "answer_key": partial}
        manifest = {**self.manifest, "sealed_answer_key_sha256": gate.digest(partial)}
        with self.assertRaisesRegex(ValueError,
                                    "^HOLD EXP053: sealed answer-key case scope"):
            self.validate(seal=seal, manifest=manifest)

    def test_63_case_packet_is_denied_even_if_keyset_matches(self):
        packet = copy.deepcopy(self.packet)
        packet["cases"].pop()
        answers = dict(list(self.answers.items())[:63])
        seal = {"packet_sha256": gate.digest(packet), "answer_key": answers}
        manifest = {"sealed_answer_key_sha256": gate.digest(answers)}
        with self.assertRaisesRegex(ValueError, "^HOLD EXP053: source packet must"):
            self.validate(packet=packet, seal=seal, manifest=manifest)

    def test_wrong_type_cases_refused(self):
        for cases in ([], {}, "invented", None):
            with self.subTest(cases=cases):
                packet = {**self.packet, "cases": cases}
                seal = {**self.seal, "packet_sha256": gate.digest(packet)}
                with self.assertRaisesRegex(ValueError, "^HOLD EXP053: source packet"):
                    self.validate(packet=packet, seal=seal)

    def test_duplicate_packet_case_identifiers_refused(self):
        packet = copy.deepcopy(self.packet)
        packet["cases"][1]["case_id"] = packet["cases"][0]["case_id"]
        seal = {**self.seal, "packet_sha256": gate.digest(packet)}
        with self.assertRaisesRegex(ValueError, "^HOLD EXP053: source packet has duplicate"):
            self.validate(packet=packet, seal=seal)

    def test_nondictionary_case_and_invalid_case_id_refused(self):
        for bad in ({"case_id": []}, None, [], "invented", {"case_id": ""},
                    {"case_id": None}):
            with self.subTest(bad=repr(bad)):
                packet = copy.deepcopy(self.packet)
                packet["cases"][4] = bad
                seal = {**self.seal, "packet_sha256": gate.digest(packet)}
                with self.assertRaisesRegex(ValueError, "^HOLD EXP053"):
                    self.validate(packet=packet, seal=seal)

    def test_extraneous_foreign_case_refused_even_with_matching_digest(self):
        answers = {**self.answers, "invented-foreign": "A"}
        seal = {**self.seal, "answer_key": answers}
        manifest = {**self.manifest, "sealed_answer_key_sha256": gate.digest(answers)}
        with self.assertRaisesRegex(ValueError, "^HOLD EXP053: sealed answer-key case scope"):
            self.validate(seal=seal, manifest=manifest)

    def test_noncanonical_answer_labels_denied(self):
        for value in ("", "AB", "Z", 1, None, True, [], {"A": "B"}):
            with self.subTest(value=repr(value)):
                answers = {**self.answers, "invented-003": value}
                seal = {**self.seal, "answer_key": answers}
                manifest = {**self.manifest, "sealed_answer_key_sha256": gate.digest(answers)}
                with self.assertRaisesRegex(ValueError,
                                            "^HOLD EXP053: sealed answer key contains"):
                    self.validate(seal=seal, manifest=manifest)

    def test_bad_seal_outer_shape_has_sanitized_hold(self):
        for seal in ({}, {"answer_key": {}}, {"packet_sha256": "0" * 64},
                     [], None, {**self.seal, "unexpected": "x"}):
            with self.subTest(seal=repr(seal)[:40]):
                with self.assertRaisesRegex(ValueError,
                                            "^HOLD EXP053: sealed answer-key record"):
                    self.validate(seal=seal)

    def test_incorrect_seal_packet_digest_denied(self):
        seal = {**self.seal, "packet_sha256": "0" * 64}
        with self.assertRaisesRegex(ValueError, "^HOLD EXP053: sealed answer-key packet"):
            self.validate(seal=seal)

    def test_nonobject_answer_map_denied_before_digest(self):
        for keys in (None, [], 5, "invented", True):
            with self.subTest(keys=keys):
                seal = {**self.seal, "answer_key": keys}
                with self.assertRaisesRegex(ValueError, "^HOLD EXP053: sealed answer-key case map"):
                    self.validate(seal=seal)

    def test_bad_manifest_seal_sha_denied(self):
        for pin in (None, 0, "", "F" * 64, "00" * 31, []):
            with self.subTest(pin=repr(pin)):
                manifest = {"sealed_answer_key_sha256": pin}
                with self.assertRaisesRegex(ValueError,
                                            "^HOLD EXP053: public manifest sealed"):
                    self.validate(manifest=manifest)

    def test_no_test_requires_reading_real_private_or_source_files(self):
        with patch.object(gate, "SOURCE", Path('/never-open-real-source-in-test')), \
             patch.object(gate, "_read_private_admission_json",
                          side_effect=AssertionError('private input forbidden')):
            self.assertEqual(len(self.validate()), 64)


if __name__ == "__main__":
    unittest.main()
