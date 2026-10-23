"""Synthetic-only deterministic completion-flow guard."""
from pathlib import Path

SOURCE = (Path(__file__).resolve().parents[1] / "research" /
          "exp053_k2_remaining_batches.sh").read_text()

def test_ordered_no_overlap_bounded_batches():
    assert "for start in 32 40 48 56;" in SOURCE
    assert "--start \"$start\" --count 8 --execute-source-backed" in SOURCE
    assert "exp053_agent_k2_summary.py --count 32" in SOURCE

def test_stop_on_error_and_readonly_private_research():
    assert "set -euo pipefail" in SOURCE
    assert "exp053_agent_three_model_compare.py --write-private" in SOURCE
    assert "EXP053_ALL_THREE_MACHINE_ONLY_COHORTS_PASS" in SOURCE
    for forbidden in ("docker compose up", "--privileged", "git push",
                      "kubectl apply", "systemctl restart", "rm -rf"):
        assert forbidden not in SOURCE
