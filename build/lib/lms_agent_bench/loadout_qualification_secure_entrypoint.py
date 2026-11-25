"""Installed qualification operator with secret-safe Hermes subprocesses."""
from __future__ import annotations

import sys
from typing import List, Optional, Sequence

from lms_agent_bench import loadout_qualification_operator as _base

_ACTIVE_API_KEY_ENV = "HERMES_BENCH_API_KEY"
_PATCHED = False


def _secure_module(module: str, *args: str) -> List[str]:
    if module != "lms_agent_bench.hermes_agent_bench":
        return [sys.executable, "-m", module, *args]
    cleaned: List[str] = []
    values = list(args)
    index = 0
    while index < len(values):
        if values[index] == "--api-key":
            index += 2
            continue
        cleaned.append(values[index])
        index += 1
    cleaned.extend(["--api-key-env", _ACTIVE_API_KEY_ENV])
    return [
        sys.executable,
        "-m",
        "lms_agent_bench.hermes_agent_secure_entrypoint",
        *cleaned,
    ]


def apply_patches() -> None:
    global _PATCHED
    if _PATCHED:
        return
    original_run = _base.run_qualification

    def secure_run(args):
        global _ACTIVE_API_KEY_ENV
        _ACTIVE_API_KEY_ENV = str(args.api_key_env)
        return original_run(args)

    _base._module = _secure_module  # noqa: SLF001
    _base.run_qualification = secure_run
    _PATCHED = True


def build_parser():
    return _base.build_parser()


def main(argv: Optional[Sequence[str]] = None) -> int:
    apply_patches()
    return _base.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
