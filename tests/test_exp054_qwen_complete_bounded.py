"""Synthetic orchestration contract; no model or private corpus loaded."""
from pathlib import Path

BODY=(Path(__file__).resolve().parents[1]/"research"/"exp054_qwen_complete_bounded.sh").read_text()


def test_8case_scopes_with_monotonic_private_audit():
    assert "set -euo pipefail" in BODY
    assert "for start in 32 40 48 56" in BODY
    assert "start+8" in BODY
    assert "--start \"$start\" --count 8 --execute-source-backed" in BODY
    assert "python3 -B research/exp054_qwen_threefoil_audit.py --count 32" in BODY
    assert "python3 -B research/exp054_qwen_threefoil_audit.py --count \"$total\" --write-private" in BODY
    assert BODY.index("exp054_qwen_threefoil_audit.py --count 32") < BODY.index("for start in")


def test_no_production_or_parallel_changes():
    for marker in ("HF_HUB_OFFLINE=1","TRANSFORMERS_OFFLINE=1",
                   "CUDA_VISIBLE_DEVICES=''","timeout 360s","nice -n 15","ionice -c3"):
        assert marker in BODY
    for forbidden in ("docker compose up","git push","--privileged",
                      "--network host","systemctl restart","rm -rf","kubectl apply",
                      "nohup ","&>","disown"):
        assert forbidden not in BODY
