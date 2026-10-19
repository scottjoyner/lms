"""Read-only local Git lineage audit tests; no remote or model operations."""

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

RESEARCH = Path(__file__).resolve().parents[1] / "research"
sys.path.insert(0, str(RESEARCH))
import exp053_branch_lineage_preflight as lineage


class DisposableGitTopology(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="exp053-git-ancestry-test-")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.env = {
            **os.environ,
            "GIT_AUTHOR_NAME": "Synthetic Test",
            "GIT_AUTHOR_EMAIL": "synthetic@example.invalid",
            "GIT_COMMITTER_NAME": "Synthetic Test",
            "GIT_COMMITTER_EMAIL": "synthetic@example.invalid",
            "GIT_TERMINAL_PROMPT": "0",
        }
        self.run_git("init", "-q", "-b", "main")
        (self.root / "public.txt").write_text("synthetic fixture\n")
        self.run_git("add", "public.txt")
        self.run_git("commit", "-qm", "synthetic initial root")
        self.base = self.run_git("rev-parse", "HEAD").stdout.strip()
        self.run_git("update-ref", lineage.MAIN_REF, self.base)
        self.run_git("update-ref", lineage.PR32_REF, self.base)

    def run_git(self, *args):
        return subprocess.run(["git", "-C", str(self.root), *args],
                              capture_output=True, text=True, check=True,
                              env=self.env, timeout=8)

    def synthetic_commit(self, name, content="synthetic"):
        target = self.root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content + "\n")
        self.run_git("add", name)
        self.run_git("commit", "-qm", f"test {name}")

    def test_connected_ref_still_cannot_authorize_integration(self):
        self.synthetic_commit("research/read_only_note.txt")
        value = lineage.inspect(self.root)
        self.assertEqual(value["lineage"], {"main": "CONNECTED", "pr32": "CONNECTED"})
        self.assertEqual(value["repository_history"], "COMPLETE")
        self.assertEqual(value["tree_delta_counts"]["main"]["A"], 1)
        self.assertEqual(value["target_pr_base_ref_status"], "MISSING")
        self.assertEqual(value["source_only_handoff_file_presence"], {"local": [], "pr32": []})
        self.assertFalse(value["integration_authorized"])
        self.assertFalse(value["scientific_execution_authorized"])
        self.assertEqual(value["decision"], "NO_GO_RESEARCH_ONLY")
        self.assertNotIn(str(self.root), json.dumps(value))

    def test_disconnected_orphan_root_detected_without_mutations(self):
        self.run_git("checkout", "--orphan", "research-orphan")
        (self.root / "public.txt").unlink(missing_ok=True)
        self.run_git("add", "-A")
        self.synthetic_commit("research/orphan.txt")
        before = self.run_git("rev-parse", "HEAD").stdout.strip()
        result = lineage.inspect(self.root)
        after = self.run_git("rev-parse", "HEAD").stdout.strip()
        self.assertEqual(before, after)
        self.assertEqual(result["lineage"], {"main": "DISCONNECTED", "pr32": "DISCONNECTED"})
        self.assertEqual(result["repository_history"], "COMPLETE")
        self.assertEqual(result["references"]["origin_main"], "FOUND")
        self.assertGreater(result["tree_delta_counts"]["pr32"]["A"], 0)
        self.assertGreater(result["tree_delta_counts"]["pr32"]["D"], 0)
        self.assertFalse(result["integration_authorized"])

    def test_shallow_history_with_hidden_parent_is_unverified_not_disconnected(self):
        # Reproduce Fleet Commander's real shallow clone failure mode in an
        # isolated, disposable Git fixture. Git retains the object's parent
        # but refuses traversal across the marked shallow boundary.
        self.synthetic_commit("research/shallow_child.txt")
        head = self.run_git("rev-parse", "HEAD").stdout.strip()
        shallow = Path(self.run_git("rev-parse", "--git-path", "shallow").stdout.strip())
        if not shallow.is_absolute():
            shallow = self.root / shallow
        shallow.write_text(head + "\n")
        self.assertEqual(self.run_git("rev-parse", "--is-shallow-repository").stdout.strip(),
                         "true")
        result = lineage.inspect(self.root)
        self.assertEqual(result["repository_history"], "SHALLOW")
        self.assertEqual(result["lineage"], {
            "main": "UNVERIFIED_SHALLOW_HISTORY",
            "pr32": "UNVERIFIED_SHALLOW_HISTORY",
        })
        self.assertFalse(result["integration_authorized"])
        self.assertFalse(result["scientific_execution_authorized"])

    def test_shallow_history_with_visible_merge_base_still_connected(self):
        # Even in a shallow repo, a visible common base is positive evidence.
        self.synthetic_commit("research/shallow_child.txt")
        head = self.run_git("rev-parse", "HEAD").stdout.strip()
        shallow = Path(self.run_git("rev-parse", "--git-path", "shallow").stdout.strip())
        if not shallow.is_absolute():
            shallow = self.root / shallow
        shallow.write_text(self.base + "\n")
        self.run_git("update-ref", lineage.PR32_REF, head)
        result = lineage.inspect(self.root)
        self.assertEqual(result["repository_history"], "SHALLOW")
        self.assertEqual(result["lineage"]["pr32"], "CONNECTED")
        self.assertEqual(result["lineage"]["main"], "CONNECTED")
        self.assertFalse(result["integration_authorized"])

    def test_missing_pr_ref_stays_unresolved_not_connected(self):
        self.run_git("update-ref", "-d", lineage.PR32_REF)
        data = lineage.inspect(self.root)
        self.assertEqual(data["references"]["pr32"], "MISSING")
        self.assertEqual(data["lineage"]["pr32"], "UNRESOLVED")
        self.assertIsNone(data["tree_delta_counts"]["pr32"])
        self.assertEqual(data["source_only_handoff_file_presence"]["pr32"], [])
        self.assertFalse(data["integration_authorized"])

    def test_missing_main_ref_stays_unresolved(self):
        self.run_git("update-ref", "-d", lineage.MAIN_REF)
        data = lineage.inspect(self.root)
        self.assertEqual(data["references"]["origin_main"], "MISSING")
        self.assertEqual(data["lineage"]["main"], "UNRESOLVED")
        self.assertIsNone(data["tree_delta_counts"]["main"])

    def test_local_handoff_presence_is_read_only(self):
        for name in lineage.HANDOFF:
            self.synthetic_commit(name)
        value = lineage.inspect(self.root)
        self.assertEqual(value["source_only_handoff_file_presence"]["local"],
                         list(lineage.HANDOFF))
        self.assertEqual(value["source_only_handoff_file_presence"]["pr32"], [])
        self.assertFalse(value["integration_authorized"])

    def test_target_pr_base_status_is_not_interpreted_as_authority(self):
        self.run_git("update-ref", lineage.PR32_BASE_REF, self.base)
        value = lineage.inspect(self.root)
        self.assertEqual(value["target_pr_base_ref_status"], "FOUND")
        self.assertFalse(value["integration_authorized"])

    def test_dirty_tracked_worktree_counts_without_listing_paths(self):
        (self.root / "public.txt").write_text("private local dirty worktree body\n")
        value = lineage.inspect(self.root)
        self.assertEqual(value["tracked_worktree_modification_count"], 1)
        self.assertNotIn("private", json.dumps(value))
        self.assertNotIn("public.txt", json.dumps(value))

    def test_command_line_never_returns_success_even_for_connected(self):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = lineage.main(["--repo", str(self.root)])
        self.assertEqual(code, 2)
        answer = json.loads(buffer.getvalue())
        self.assertEqual(answer["decision"], "NO_GO_RESEARCH_ONLY")
        self.assertFalse(answer["scientific_execution_authorized"])
        self.assertFalse(answer["integration_authorized"])
        self.assertNotIn(str(self.root), buffer.getvalue())

    def test_path_validation_prevents_running_on_ambiguous_paths(self):
        link = self.root.parent / (self.root.name + "-link")
        link.symlink_to(self.root, target_is_directory=True)
        self.addCleanup(link.unlink)
        for arg in (Path("relative-path"), link, self.root / "absent"):
            with self.subTest(case=str(arg)), self.assertRaises(lineage.Hold):
                lineage.inspect(arg)

    def test_corrupt_local_git_repository_denied_without_error_leakage(self):
        with tempfile.TemporaryDirectory(prefix="exp053-no-git-repo-") as tmp:
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                rc = lineage.main(["--repo", tmp])
            self.assertEqual(rc, 2)
            result = json.loads(buffer.getvalue())
            self.assertEqual(result["status"], "UNVERIFIABLE_REPOSITORY_LINEAGE")
            self.assertFalse(result["integration_authorized"])
            self.assertNotIn(tmp, buffer.getvalue())

    def test_gits_used_by_inspection_are_bounded_read_only(self):
        old = lineage._git
        seen = []
        def check(repo, *args, **kw):
            seen.append(args)
            self.assertIn(args[0], {"rev-parse", "merge-base", "diff-tree", "status", "cat-file"})
            self.assertNotIn("push", args)
            self.assertNotIn("fetch", args)
            return old(repo, *args, **kw)
        with patch.object(lineage, "_git", side_effect=check):
            result = lineage.inspect(self.root)
        self.assertGreater(len(seen), 3)
        self.assertFalse(result["integration_authorized"])


if __name__ == "__main__":
    unittest.main()
