#!/usr/bin/env python3
"""Paired native Hermes tool-presentation research with fresh scratch graphs.

No deployment, production Neo4j, real credentials, or model promotion.
One original native bridge and one direct MCP condition, in fixed order.
Each trial gets an exact isolated Docker ID and teardown before the next.
No promotion: two trials can never satisfy the three-trial acceptance gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from urllib.error import URLError
from urllib.request import urlopen

from lms_agent_bench import hermes_agent_bench as bench
from lms_agent_bench.hermes_agent_common import load_json, validate_suite
from lms_agent_bench.hermes_agent_secure_entrypoint import _safe_trial_command
from lms_agent_bench.hermes_native_provenance import audit_native_trial
from lms_agent_bench.hermes_scratch_replay import ScratchHermesArchive
from lms_agent_bench.model_loadout import validate_manifest

ROOT = Path(__file__).resolve().parents[1]
HERMES = Path("/home/scott/git/hermes-agent")
PYTHON = HERMES / "venv/bin/python"
SERVER = Path("/home/scott/src/llama.cpp-bonsai2/build/bin/llama-server")
MODEL = Path(
    "/home/scott/fleet-data/model-evaluation/candidates/"
    "lfm25-24b-a2b-q4km/LFM2.5-24B-A2B-Q4_K_M.gguf"
)
LOADOUT = Path(
    "/home/scott/knowledge/50-Research/model-evaluation/results/"
    "lfm25-24b-64k-parity-2026-10-14/native-preserve-loadout.json"
)
SUITE = ROOT / "src/lms_agent_bench/benchmarks/hermes_agent_suite.v1.json"
ENTRY = ROOT / "scripts/hermes_native_tool_presentation_entrypoint.py"
ENDPOINT = "http://127.0.0.1:19247/v1"
PORT = 19247
EXPECTED_SHA = "d3334405b96228eb9908cdebb5c6194908f2da64282bda852afc3a3910c64e7d"


def _model_digest() -> str:
    digest = hashlib.sha256()
    with MODEL.open("rb") as reader:
        while data := reader.read(4 * 1024 * 1024):
            digest.update(data)
    return digest.hexdigest()


def _port_free():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        return probe.connect_ex(("127.0.0.1", PORT)) != 0


def _ready():
    try:
        with urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2) as response:
            return response.status == 200
    except (OSError, URLError):
        return False


def _trial_command(args, **kwargs):
    command = _safe_trial_command(args, **kwargs)
    if Path(command[1]).name != "hermes_agent_secure_entrypoint.py":
        raise RuntimeError("unsafe noncanonical trial command")
    command[1] = str(ENTRY)
    return command


def _verify_exact_container_removed(container_id: str) -> dict:
    """Fail closed if Docker is down; an arbitrary nonzero inspect is not proof."""
    if not re.fullmatch(r"[0-9a-f]{64}", container_id):
        raise RuntimeError("invalid exact owned research container ID")
    healthy = subprocess.run(
        ["docker", "info", "--format", "{{.ServerVersion}}"],
        text=True, capture_output=True, timeout=12, check=False,
    )
    if healthy.returncode != 0 or not healthy.stdout.strip():
        raise RuntimeError("Docker daemon unhealthy; teardown status UNKNOWN")
    inspected = subprocess.run(
        ["docker", "inspect", "--format", "{{.Id}}", container_id],
        text=True, capture_output=True, timeout=12, check=False,
    )
    expected = f"error: no such object: {container_id}"
    if (inspected.returncode != 1 or inspected.stdout.strip()
            or inspected.stderr.strip().lower() != expected):
        raise RuntimeError("exact owned research container absence NOT established")
    return {"docker_daemon_healthy": True, "docker_inspect_exit_code": 1,
            "exact_container_absent": True}


def _research_three_trial_accepted(state: dict) -> bool:
    """Evidence-only gate. Never authorizes release, deployment or promotion."""
    identities = state.get("owned_container_ids", [])
    trials = state.get("trials", [])
    return (len(trials) == 3
            and len(identities) == 3
            and len(set(identities)) == 3
            and state.get("native_correlated_count") == 3
            and state.get("engine_provenance_count") == 3
            and state.get("model_server_removed") is True
            and all(t.get("teardown_verified") is True
                    and t.get("exact_container_removed") is True
                    and t.get("native_complete_source_provenance") is True
                    for t in trials))


def _die(_signal, _frame):
    raise RuntimeError("native research interrupted; attempt bounded teardown")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--confirm-research-only", action="store_true")
    parser.add_argument("--trial-timeout", type=float, default=115)
    args = parser.parse_args(argv)
    if not args.confirm_research_only:
        parser.error("explicit research-only opt-in is required")
    if not 60 <= args.trial_timeout <= 180:
        parser.error("trial timeout must be in [60,180] seconds")
    for path in (HERMES / "run_agent.py", PYTHON, SERVER, MODEL, LOADOUT, SUITE, ENTRY):
        if not path.is_file():
            raise RuntimeError(f"required pinned research input unavailable: {path}")
    if not _port_free():
        raise RuntimeError("research loopback inference port already in use")
    if _model_digest() != EXPECTED_SHA:
        raise RuntimeError("pinned Liquid candidate GGUF digest changed")
    source = ROOT / "src"
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    suite = validate_suite(load_json(SUITE))
    selected = [x for x in suite["cases"]
                if x["case_key"] == "mcp_tool_distractor_selection"]
    if len(selected) != 1:
        raise RuntimeError("canonical research case not unique")
    case = selected[0]
    loadout = validate_manifest(load_json(LOADOUT))
    if loadout["model"]["id"] != "liquid-24b-parity-candidate":
        raise RuntimeError("noncanonical model alias")
    state = {
        "research_only": True, "release_authorized": False,
        "qualification_authorized": False,
        "model_sha256": EXPECTED_SHA,
        "run_count_expected": 2, "trials": [],
        "research_modes": ["bridge", "direct"],
        "paired_request_envelope_identical_except_tools": False,
        "owned_container_ids": [],
        "model_server_removed": False,
    }
    pinned = output / "private-native-system-preamble.txt"
    previous = signal.signal(signal.SIGTERM, _die)
    server_process = None
    bench.trial_command = _trial_command
    root = output / "trial-roots"
    root.mkdir()
    opts = SimpleNamespace(
        hermes_python=str(PYTHON), hermes_repo=str(HERMES),
        endpoint=ENDPOINT, api_key="local-benchmark",
        api_key_env="HERMES_BENCH_API_KEY", timeout_seconds=args.trial_timeout,
    )
    try:
        env = {
            **os.environ,
            "PYTHONPATH": f"{source}:{HERMES}",
            "LMS_HERMES_RESEARCH_ROOT": str(root),
            "LMS_HERMES_NATIVE_PROVENANCE_RESEARCH": "1",
            "NO_COLOR": "1", "TERM": "dumb",
            "LMS_HERMES_PINNED_PREAMBLE_FILE": str(pinned),
        }
        command = [
            str(SERVER), "--model", str(MODEL),
            "--alias", "liquid-24b-parity-candidate",
            "--host", "127.0.0.1", "--port", str(PORT),
            "--ctx-size", "65536", "--parallel", "1",
            "--threads", "4", "--n-gpu-layers", "0",
        ]
        with (output / "llama-stdout.log").open("w") as so, (
            output / "llama-stderr.log"
        ).open("w") as se:
            server_process = subprocess.Popen(
                command, stdout=so, stderr=se,
                start_new_session=True, env=env,
            )
            for _ in range(120):
                if server_process.poll() is not None:
                    raise RuntimeError("isolated inference process exited before ready")
                if _ready():
                    break
                time.sleep(1)
            else:
                raise RuntimeError("isolated research model failed 120s ready limit")
            state["server_ready"] = True
            for index, mode in enumerate(("bridge", "direct"), start=1):
                with ScratchHermesArchive() as scratch:
                    container_id = scratch.container_id
                    if container_id in state["owned_container_ids"]:
                        raise RuntimeError("duplicate research container identity")
                    state["owned_container_ids"].append(container_id)
                    os.environ.update(env)
                    os.environ["LFM25_DISPOSABLE_NEO4J_ID"] = scratch.container_id
                    os.environ["LMS_HERMES_TOOL_PRESENTATION_RESEARCH"] = mode
                    trial = bench.run_one_trial(
                        opts, case, loadout, index, root,
                        {str(t) for t in suite.get("prohibited_tools", [])},
                    )
                    container_id = scratch.container_id
                if not scratch.destroyed:
                    raise RuntimeError("per-trial scratch teardown unverified")
                teardown = _verify_exact_container_removed(container_id)
                folder = root / case["case_key"] / f"trial-{index:02d}"
                audit = audit_native_trial(
                    folder, expected_container_id=container_id,
                )
                audit["scratch_container_id"] = container_id
                audit["teardown_receipt"] = teardown
                audit["teardown_verified"] = True
                audit["exact_container_removed"] = True
                audit["runner_valid"] = bool(trial.get("valid"))
                audit["research_tool_presentation_mode"] = mode
                state["trials"].append(audit)
                (folder / "native-independent-provenance.json").write_text(
                    json.dumps(audit, sort_keys=True, indent=2) + "\n"
                )
                print(
                    "TRIAL", index, "native_correlated",
                    audit["native_transcript_http_correlated"],
                    "valid", audit["runner_valid"], "teardown", True, flush=True,
                )
    except Exception as exc:
        state["denial"] = str(exc)
    finally:
        signal.signal(signal.SIGTERM, previous)
        if server_process is not None:
            try:
                if server_process.poll() is None:
                    server_process.terminate()
                    try:
                        server_process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        server_process.kill()
                        server_process.wait(timeout=10)
                state["model_server_removed"] = server_process.poll() is not None
            except Exception as exc:
                state["model_server_teardown_denial"] = str(exc)
        state["native_correlated_count"] = sum(
            item.get("native_transcript_http_correlated") is True
            for item in state["trials"]
        )
        state["engine_provenance_count"] = sum(
            item.get("native_complete_source_provenance") is True
            for item in state["trials"]
        )
        if pinned.exists():
            # The native system text is private scratch state, NOT an archive.
            state["pinned_system_bytes"] = pinned.stat().st_size
            state["private_preamble_destroyed"] = False
            pinned.unlink()
            state["private_preamble_destroyed"] = not pinned.exists()
        try:
            if len(state["trials"]) == 2:
                records = []
                for index, mode in enumerate(("bridge", "direct"), 1):
                    folder = root / case["case_key"] / f"trial-{index:02d}"
                    wire_path = folder / "fixture-state" / "model-http-receipts.jsonl"
                    receipt_path = folder / "fixture-state" / "presentation-request-receipts.jsonl"
                    if not wire_path.is_file() or not receipt_path.is_file():
                        raise RuntimeError("missing real model wire or presentation evidence")
                    wire = [json.loads(line) for line in wire_path.read_text().splitlines()]
                    receipts = [json.loads(line) for line in receipt_path.read_text().splitlines()]
                    if not wire or not receipts or "wire_surface" not in wire[0]:
                        raise RuntimeError("incomplete physical presentation receipts")
                    records.append({"mode": mode, "wire": wire[0]["wire_surface"],
                                    "constructor": receipts[0],
                                    "request_sha256": wire[0]["request_sha256"],
                                    "model": wire[0]["model"]})
                a, b = records
                bridge_names = {"tool_search", "tool_describe", "tool_call"}
                direct_names = {"mcp__lms_benchmark__" + t for t in (
                    "graph_schema", "graph_query", "workspace_read", "workspace_write",
                    "workspace_patch", "workspace_run_tests", "weather_lookup",
                    "send_email", "graph_delete_all")}
                same = all(a["wire"].get(key) == b["wire"].get(key)
                           for key in ("system_content_sha256", "user_content_sha256",
                                       "generation", "all_other_options_sha256",
                                       "model_message_roles"))
                same &= a["model"] == b["model"]
                same &= a["wire"]["model_message_roles"] == ["system", "user"]
                same &= b["wire"]["model_message_roles"] == ["system", "user"]
                same &= a["wire"].get("advertised_tool_count") == 3
                same &= b["wire"].get("advertised_tool_count") == 9
                same &= {x["name"] for x in a["wire"]["advertised_tool_contracts"]} == bridge_names
                same &= {x["name"] for x in b["wire"]["advertised_tool_contracts"]} == direct_names
                same &= a["constructor"].get("pinned_effective_system_sha256") == b["constructor"].get("pinned_effective_system_sha256")
                state["paired_request_envelope_identical_except_tools"] = bool(same)
                state["first_request_comparison"] = records
        except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
            state["pair_denial"] = str(exc)
            state["paired_request_envelope_identical_except_tools"] = False
        state["all_three_qualified"] = False  # two-case ablation NEVER qualifies
        state["research_pair_only"] = True
        state["qualification_authorized"] = False
        state["release_authorized"] = False
        (output / "native-research-summary.json").write_text(
            json.dumps(state, sort_keys=True, indent=2) + "\n"
        )
        print("RESEARCH_SUMMARY", json.dumps({
            "trials": len(state["trials"]), "native_correlated": state["native_correlated_count"],
            "model_server_removed": state["model_server_removed"],
            "all_three_qualified": state["all_three_qualified"],
            "same_except_tools": state["paired_request_envelope_identical_except_tools"],
            "engine_provenance_count": state["engine_provenance_count"],
            "denial": state.get("denial"),
        }), flush=True)
    return 3  # 2-condition ablation cannot be 3-trial model acceptance


if __name__ == "__main__":
    sys.exit(main())
