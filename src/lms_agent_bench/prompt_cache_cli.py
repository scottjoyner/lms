"""CLI for the record-only prompt-prefix/KV artifact registry."""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from lms_agent_bench.hermes_agent_common import load_json, write_json
from lms_agent_bench.prompt_cache_identity import (
    DEFAULT_BLOCK_SIZE,
    build_compatibility_manifest,
    require_sha256,
)
from lms_agent_bench.prompt_cache_registry import (
    REGISTRY_SCHEMA_VERSION,
    SENSITIVITIES,
    PromptCacheRecorder,
    SQLitePromptCacheRegistry,
)
from lms_agent_bench.prompt_cache_store import ContentAddressedArtifactStore


def _add_identity(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--loadout", type=Path, required=True)
    parser.add_argument("--tokenizer-sha256", required=True)
    parser.add_argument("--chat-template-sha256", required=True)
    parser.add_argument("--system-prompt-sha256", required=True)
    parser.add_argument("--tool-schema-sha256", required=True)
    parser.add_argument("--engine-serialization-abi", required=True)
    parser.add_argument("--adapter-sha256")
    parser.add_argument("--multimodal-encoder-sha256")
    parser.add_argument("--preprocessing-sha256")


def _manifest(args: argparse.Namespace) -> Dict[str, Any]:
    return build_compatibility_manifest(
        load_json(args.loadout),
        tokenizer_sha256=args.tokenizer_sha256,
        chat_template_sha256=args.chat_template_sha256,
        system_prompt_sha256=args.system_prompt_sha256,
        tool_schema_sha256=args.tool_schema_sha256,
        engine_serialization_abi=args.engine_serialization_abi,
        adapter_sha256=args.adapter_sha256,
        multimodal_encoder_sha256=args.multimodal_encoder_sha256,
        preprocessing_sha256=args.preprocessing_sha256,
    )


def _tokens(path: Path) -> list[int]:
    value = load_json(path)
    if isinstance(value, Mapping):
        value = value.get("token_ids")
    if not isinstance(value, list):
        raise ValueError("token ID file must be a list or {'token_ids': [...]}")
    return [int(item) for item in value]


def _recorder(args: argparse.Namespace) -> PromptCacheRecorder:
    root = Path(args.root).expanduser()
    return PromptCacheRecorder(
        SQLitePromptCacheRegistry(root / "registry.sqlite3"),
        ContentAddressedArtifactStore(root),
    )


def _emit(value: Mapping[str, Any], output: Optional[Path] = None) -> None:
    if output is not None:
        write_json(output, value)
    print(json.dumps(value, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lms-prompt-cache",
        description="Record-only exact prompt-prefix and KV artifact registry",
    )
    parser.add_argument(
        "--root",
        default=os.environ.get("LMS_PROMPT_CACHE_ROOT", "~/.cache/lms-kv"),
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("init", help="Initialize the local registry")

    observe = commands.add_parser("observe", help="Record a request and cache candidate")
    _add_identity(observe)
    observe.add_argument("--token-ids", type=Path, required=True)
    observe.add_argument("--namespace", required=True)
    observe.add_argument("--node-id", required=True)
    observe.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    observe.add_argument("--estimated-prefill-ms-per-token", type=float)
    observe.add_argument("--out", type=Path)

    register = commands.add_parser(
        "register-artifact", help="Register an opaque engine-native artifact"
    )
    _add_identity(register)
    register.add_argument("--token-ids", type=Path, required=True)
    register.add_argument("--artifact", type=Path, required=True)
    register.add_argument("--namespace", required=True)
    register.add_argument("--node-id", required=True)
    register.add_argument("--serialization-format", required=True)
    register.add_argument("--serialization-version", required=True)
    register.add_argument(
        "--sensitivity", choices=sorted(SENSITIVITIES), default="private"
    )
    register.add_argument("--expires-at-utc")
    register.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    register.add_argument("--out", type=Path)

    stats = commands.add_parser("stats", help="Report metadata and observations")
    stats.add_argument("--namespace")
    stats.add_argument("--out", type=Path)

    verify = commands.add_parser("verify", help="Verify a local payload SHA-256")
    verify.add_argument("--payload-sha256", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    recorder = _recorder(args)
    if args.command == "init":
        recorder.initialize()
        _emit({"schema_version": REGISTRY_SCHEMA_VERSION,
               "root": str(Path(args.root).expanduser()), "mode": "record_only",
               "admission": {"admitted": False}})
        return 0
    if args.command == "observe":
        _emit(recorder.observe_request(
            _manifest(args), _tokens(args.token_ids), namespace=args.namespace,
            node_id=args.node_id, block_size=args.block_size,
            estimated_prefill_ms_per_token=args.estimated_prefill_ms_per_token), args.out)
        return 0
    if args.command == "register-artifact":
        artifact = recorder.register_local_artifact(
            _manifest(args), _tokens(args.token_ids), args.artifact,
            namespace=args.namespace, node_id=args.node_id,
            serialization_format=args.serialization_format,
            serialization_version=args.serialization_version,
            sensitivity=args.sensitivity, expires_at_utc=args.expires_at_utc,
            block_size=args.block_size)
        _emit({"schema_version": REGISTRY_SCHEMA_VERSION,
               "artifact": dataclasses.asdict(artifact), "mode": "record_only",
               "restoration_enabled": False, "admission": {"admitted": False}}, args.out)
        return 0
    if args.command == "stats":
        _emit(recorder.registry.stats(namespace=args.namespace), args.out)
        return 0
    if args.command == "verify":
        valid = recorder.store.verify(args.payload_sha256)
        _emit({"payload_sha256": require_sha256(args.payload_sha256, "payload_sha256"),
               "valid": valid})
        return 0 if valid else 1
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
