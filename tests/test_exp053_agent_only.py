"""No human identity, dataset access or model forwards in these synthetic tests."""
import copy
import contextlib
import io
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research"))
import exp053_agent_only as agent
import exp053_race_review as src


class MachineOnly(unittest.TestCase):
    def setUp(self):
        self.packet = {
            "schema": "exp053-packet-v1", "source_sha256": src.SOURCE_SHA,
            "protocol_sha256": src.PROTOCOL_SHA,
            "license": "RACE: non-commercial research only; do not redistribute",
            "cases": [
                {"case_id": f"case-{i:03}", "article": f"Article source {i}",
                 "question": f"Question source {i}",
                 "options": {"A": "Yes", "B": "No", "C": "Maybe", "D": "Unknown"}}
                for i in range(src.TARGET)
            ],
        }
        self.answers = {x["case_id"]: "B" for x in self.packet["cases"]}
        self.seal = {"packet_sha256": src.digest(self.packet), "answer_key": self.answers}
        self.manifest = {
            "schema": "exp053-public-manifest-v1",
            "source_sha256": src.SOURCE_SHA,
            "protocol_sha256": src.PROTOCOL_SHA,
            "evaluation_admitted": False,
            "packet_sha256": src.digest(self.packet),
            "sealed_answer_key_sha256": src.digest(self.answers),
            "rows": [
                {"case_id": x["case_id"],
                 "article_sha256": src.digest(x["article"].encode()),
                 "question_sha256": src.digest(x["question"].encode()),
                 "options_sha256": src.digest(src.canonical([x["options"][l] for l in src.LETTERS]))}
                for x in self.packet["cases"]
            ],
        }

    def patches(self):
        def fake_read(path):
            return copy.deepcopy(self.packet if Path(path).name == "packet.json" else self.seal)
        return [
            patch.object(src, "_read_pinned_public_manifest", return_value=self.manifest),
            patch.object(src, "_read_private_admission_json", side_effect=fake_read),
            patch.object(src, "SOURCE", Path(__file__)),
            patch.object(src, "PROTOCOL", Path(src.__file__)),
            patch.object(src, "filehash", side_effect=lambda _path:
                         src.SOURCE_SHA if _path == src.SOURCE else src.PROTOCOL_SHA),
        ]

    def with_patches(self, fn):
        with contextlib.ExitStack() as st:
            for p in self.patches():
                st.enter_context(p)
            return fn()

    def test_proxy_has_no_human_reviewers_and_exact_scope(self):
        receipt = self.with_patches(lambda: agent.machine_receipt(Path("/safe-private")))
        self.assertEqual(receipt["schema"], agent.SCHEMA)
        self.assertFalse(receipt["human_approval"])
        self.assertFalse(receipt["evaluation_admitted_as_human_validated"])
        self.assertEqual(receipt["human_reviewers"], 0)
        self.assertEqual(receipt["model_forward_passes"], 0)
        self.assertEqual(len(receipt["accepted"]), 64)
        self.assertTrue(all(v["supported_exam_key"] == "B" and
                            v["other_exam_option_proxy"] != "B"
                            for v in receipt["accepted"].values()))
        self.assertEqual(receipt, self.with_patches(lambda: agent.machine_receipt(Path("/safe-private"))))

    def test_tampered_private_passage_rejected(self):
        self.packet["cases"][0]["article"] += " TAMPER"
        self.manifest["packet_sha256"] = src.digest(self.packet)
        self.seal["packet_sha256"] = src.digest(self.packet)
        with self.assertRaisesRegex(ValueError, "contents not bound"):
            self.with_patches(lambda: agent.machine_receipt(Path("/safe-private")))

    def test_packet_license_restriction_rejected_if_missing(self):
        self.packet["license"] = "unrestricted"
        self.manifest["packet_sha256"] = src.digest(self.packet)
        self.seal["packet_sha256"] = src.digest(self.packet)
        with self.assertRaisesRegex(ValueError, "license"):
            self.with_patches(lambda: agent.machine_receipt(Path("/safe-private")))

    def test_duplicate_wrong_options_rejected(self):
        self.packet["cases"][0]["options"]["C"] = "No"
        self.manifest["packet_sha256"] = src.digest(self.packet)
        self.seal["packet_sha256"] = src.digest(self.packet)
        with self.assertRaisesRegex(ValueError, "ambiguous options"):
            self.with_patches(lambda: agent.machine_receipt(Path("/safe-private")))

    def test_modified_answer_key_rejected(self):
        self.answers["case-001"] = "A"
        with self.assertRaisesRegex(ValueError, "answer key differs"):
            self.with_patches(lambda: agent.machine_receipt(Path("/safe-private")))

    def test_public_only_mode_never_accesses_private_inputs(self):
        with patch.object(src, "_read_pinned_public_manifest", return_value=self.manifest), \
             patch.object(src, "_read_private_admission_json", side_effect=AssertionError("PRIVATE READ FORBIDDEN")):
            with io.StringIO() as stream, contextlib.redirect_stdout(stream):
                self.assertEqual(agent.main([]), 0)
                record = json.loads(stream.getvalue())
        self.assertFalse(record["human_review_required"])
        self.assertFalse(record["human_approval"])
        self.assertEqual(record["model_forward_passes"], 0)
        self.assertEqual(record["status"], "SOURCE_ONLY_PREFLIGHT_VALID_NO_MODEL_OR_HUMAN")

    def test_source_manifest_invalid_refused_without_private_read(self):
        self.manifest["rows"] = self.manifest["rows"][:-1]
        with patch.object(src, "_read_pinned_public_manifest", return_value=self.manifest), \
             patch.object(src, "_read_private_admission_json", side_effect=AssertionError("NO PRIVATE READ")):
            with self.assertRaisesRegex(ValueError, "cohort size"):
                agent.machine_receipt(Path("/safe-private"))

    def test_unpaired_private_parameters_are_rejected(self):
        with self.assertRaises(SystemExit):
            agent.main(["--private-dir", "/private-path"])


if __name__ == "__main__":
    unittest.main()
