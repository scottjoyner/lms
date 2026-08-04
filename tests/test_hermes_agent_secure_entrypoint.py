from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from lms_agent_bench import hermes_agent_secure_entrypoint as secure
from lms_agent_bench import loadout_qualification_secure_entrypoint as qualification


def namespace(**changes):
    values = {
        "hermes_python": "/usr/bin/python3",
        "hermes_repo": "/srv/hermes",
        "endpoint": "http://127.0.0.1:8080/v1",
        "api_key": "local-benchmark",
        "api_key_env": "HERMES_SECRET",
    }
    values.update(changes)
    return argparse.Namespace(**values)


def test_trial_command_contains_only_environment_name(tmp_path):
    args = namespace(api_key="super-secret-api-key")
    command = secure._safe_trial_command(
        args,
        case_path=tmp_path / "case.json",
        loadout_path=tmp_path / "loadout.json",
        result_path=tmp_path / "result.json",
        home=tmp_path / "home",
        workspace=tmp_path / "workspace",
        state_dir=tmp_path / "state",
    )
    joined = " ".join(command)
    assert "super-secret-api-key" not in joined
    assert "--api-key-env HERMES_SECRET" in joined
    assert Path(command[1]).name == "hermes_agent_secure_entrypoint.py"


def test_run_key_is_resolved_from_environment(monkeypatch):
    args = namespace()
    monkeypatch.setenv("HERMES_SECRET", "super-secret-api-key")
    secure._resolve_run_key(args)
    assert args.api_key == "super-secret-api-key"


def test_explicit_nondefault_command_line_key_is_rejected(monkeypatch):
    monkeypatch.delenv("HERMES_SECRET", raising=False)
    args = namespace(api_key="do-not-put-this-on-command-line")
    with pytest.raises(ValueError, match="--api-key-env"):
        secure._resolve_run_key(args)


def test_hidden_trial_defaults_to_nonsecret_local_key(monkeypatch):
    monkeypatch.delenv("HERMES_SECRET", raising=False)
    args = namespace(api_key=None)
    secure._resolve_trial_key(args)
    assert args.api_key == "local-benchmark"


def test_secure_parser_does_not_require_hidden_api_key_argument():
    parser = secure.build_parser()
    parsed = parser.parse_args(
        [
            "_trial",
            "--hermes-repo",
            "/srv/hermes",
            "--case",
            "case.json",
            "--loadout",
            "loadout.json",
            "--endpoint",
            "http://127.0.0.1:8080/v1",
            "--api-key-env",
            "HERMES_SECRET",
            "--result",
            "result.json",
            "--hermes-home",
            "home",
            "--workspace",
            "workspace",
            "--state-dir",
            "state",
        ]
    )
    assert parsed.api_key is None
    assert parsed.api_key_env == "HERMES_SECRET"


def test_secure_qualification_command_rewrites_hermes_module():
    qualification._ACTIVE_API_KEY_ENV = "QUALIFICATION_SECRET"
    command = qualification._secure_module(
        "lms_agent_bench.hermes_agent_bench",
        "run",
        "--endpoint",
        "http://127.0.0.1:8080/v1",
        "--api-key",
        "local-benchmark",
        "--out",
        "report.json",
    )
    joined = " ".join(command)
    assert "lms_agent_bench.hermes_agent_secure_entrypoint" in joined
    assert "--api-key local-benchmark" not in joined
    assert "--api-key-env QUALIFICATION_SECRET" in joined


def test_nonhermes_qualification_modules_are_unchanged():
    command = qualification._secure_module(
        "lms_agent_bench.loadout_qualification", "verify"
    )
    assert command == [
        command[0],
        "-m",
        "lms_agent_bench.loadout_qualification",
        "verify",
    ]
