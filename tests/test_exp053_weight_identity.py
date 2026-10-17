"""Synthetic unit checks for the EXP053 read-only checkpoint identity preflight."""
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "research"
sys.path.insert(0, str(R))
import exp053_weight_identity as identity


class CheckpointIdentityTests(unittest.TestCase):
    def test_pinned_protocol(self):
        self.assertEqual(identity.filehash(identity.SPEC), identity.SPEC_SHA)
        self.assertEqual(identity.filehash(identity.TOKENS_REPORT),
                         identity.TOKENS_REPORT_SHA)

    def test_streaming_synthetic_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            content = b"research fixture" * 512
            file = folder / "model.safetensors"
            file.write_bytes(content)
            receipt = identity.stream_digest(file, folder)
            self.assertEqual(receipt["sha256"], hashlib.sha256(content).hexdigest())
            self.assertEqual(receipt["bytes"], len(content))

    def test_no_unexpected_shards(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "model.safetensors.index.json").write_text(json.dumps(
                {"weight_map": {"a": "one.safetensors", "b": "two.safetensors"}}
            ))
            (folder / "one.safetensors").write_bytes(b"A" * 8192)
            with self.assertRaisesRegex(ValueError, "HOLD"):
                identity.source_files("K2-Horizon-3.7B", folder)
            (folder / "two.safetensors").write_bytes(b"B" * 8192)
            shards, digest = identity.source_files("K2-Horizon-3.7B", folder)
            self.assertEqual([f.name for f in shards], ["one.safetensors", "two.safetensors"])
            self.assertEqual(digest, identity.filehash(folder / "model.safetensors.index.json"))
            (folder / "extra.safetensors").write_bytes(b"C" * 8192)
            with self.assertRaisesRegex(ValueError, "HOLD"):
                identity.source_files("K2-Horizon-3.7B", folder)

    def test_single_file_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            file = folder / "model.safetensors"
            file.write_bytes(b"Q" * 8192)
            self.assertEqual(identity.source_files("MiniCPM-V-4.7-1B", folder)[0], [file])
            (folder / "unexpected.safetensors").write_bytes(b"X" * 8192)
            with self.assertRaisesRegex(ValueError, "HOLD"):
                identity.source_files("MiniCPM-V-4.7-1B", folder)

    def test_short_file_denied(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            file = folder / "model.safetensors"
            file.write_bytes(b"tiny")
            with self.assertRaisesRegex(ValueError, "HOLD"):
                identity.stream_digest(file, folder)

    def test_shared_hub_content_addressed_cache_with_strict_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            hub = Path(directory) / "hub"
            snap = hub / "models--synthetic--checkpoint" / "snapshots" / "revision"
            snap.mkdir(parents=True)
            shared = hub / "blobs" / "ab"
            shared.mkdir(parents=True)
            content_name = "ab" + "1" * 62
            fixture = shared / content_name
            fixture.write_bytes(b"P" * 8192)
            pointer = snap / "model.safetensors"
            pointer.symlink_to(fixture)
            self.assertEqual(identity.allowed_resolved(pointer, snap), fixture)
            self.assertEqual(identity.stream_digest(pointer, snap)["bytes"], 8192)
            # The shared hub is NOT a free-for-all symlink target.
            invalid = hub / "blobs" / "ab" / ("cd" + "2" * 62)
            invalid.write_bytes(b"Q" * 8192)
            pointer.unlink()
            pointer.symlink_to(invalid)
            with self.assertRaisesRegex(ValueError, "HOLD"):
                identity.allowed_resolved(pointer, snap)
            pointer.unlink()
            outside = Path(directory) / "outside" / "model.safetensors"
            outside.parent.mkdir()
            outside.write_bytes(b"Z" * 8192)
            pointer.symlink_to(outside)
            with self.assertRaisesRegex(ValueError, "HOLD"):
                identity.allowed_resolved(pointer, snap)

    def test_schema_hash_is_canonical(self):
        x = {"model": "synthetic", "weights": ["A", "B"]}
        self.assertEqual(identity.sha(x), identity.sha({"weights": ["A", "B"],
                                                       "model": "synthetic"}))


if __name__ == "__main__":
    unittest.main()
