"""Synthetic-only security contract for K2's physical research entrypoint."""
from pathlib import Path

LAUNCHER=Path(__file__).resolve().parents[1]/"research"/"exp053_k2_isolated.sh"

def test_rootless_networkless_readonly_k2_entrypoint():
    source=LAUNCHER.read_text()
    for required in (
        "--network none", "--read-only", "--cap-drop ALL",
        "--security-opt no-new-privileges", "--pids-limit 96",
        "--memory 28g", "--memory-swap 28g", "--cpus 2",
        '--user "$(id -u):$(id -g)"',
        "$model:$model:ro", "$root:$root:ro",
        "$private:$private:ro", "$output:$output:rw",
        "HF_HUB_OFFLINE=1", "TRANSFORMERS_OFFLINE=1",
        "EXP053_K2_SANDBOX=networkless-readonly",
    ):
        assert required in source

def test_no_privileged_or_host_network_flags():
    source=LAUNCHER.read_text()
    for forbidden in ("--privileged", "--network host", "--cap-add",
                      "--device=", "--pid host", "--ipc host"):
        assert forbidden not in source

def test_pinned_corpus_cas_readonly():
    source=LAUNCHER.read_text()
    assert 'readlink -f' in source
    assert 'case "$cas_file"' in source
    assert '$cas_file:$cas_file:ro' in source
