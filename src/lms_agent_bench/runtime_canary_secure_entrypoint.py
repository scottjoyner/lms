"""Security boundary for the installed runtime canary command."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from lms_agent_bench import runtime_canary as _base

_SHELL_NAMES = {
    "sh",
    "bash",
    "dash",
    "zsh",
    "fish",
    "ksh",
    "pwsh",
    "powershell",
    "cmd.exe",
}
_ORIGINAL_LOAD_PLAN = _base.load_plan
_PATCHED = False


def _reject_shell_commands(path: Path) -> None:
    source = Path(path).expanduser()
    if source.is_symlink():
        raise ValueError("canary plan may not be a symbolic link")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("canary plan must be an object")
    commands = payload.get("commands")
    if not isinstance(commands, dict):
        raise ValueError("commands must be an object")
    for name, raw in commands.items():
        if not isinstance(raw, dict):
            raise ValueError(f"commands.{name} must be an object")
        argv = raw.get("argv")
        if not isinstance(argv, list) or not argv or not isinstance(argv[0], str):
            raise ValueError(f"commands.{name}.argv must be a nonempty string array")
        executable = Path(argv[0]).expanduser().resolve()
        if executable.name.lower() in _SHELL_NAMES:
            raise ValueError(f"commands.{name} may not invoke a shell interpreter")


def secure_load_plan(path: Path) -> Dict[str, Any]:
    _reject_shell_commands(path)
    return _ORIGINAL_LOAD_PLAN(path)


def apply_patches() -> None:
    global _PATCHED
    if _PATCHED:
        return
    _base.load_plan = secure_load_plan
    _PATCHED = True


def main(argv: Optional[Sequence[str]] = None) -> int:
    apply_patches()
    return _base.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
