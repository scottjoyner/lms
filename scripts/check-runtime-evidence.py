#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import socket
import sys
from pathlib import Path

from lms_agent_bench import runtime_identity_witness as witness


_STRONG_BINDINGS = {"proc_maps", "darwin_vmmap_lsof"}


def probe(pid: int, model_path: Path) -> dict:
    model = witness._regular_model(model_path)  # noqa: SLF001
    model_identity = witness._stat_identity(model.stat())  # noqa: SLF001
    process = witness.process_identity(pid)
    binding = witness._process_references_model(  # noqa: SLF001
        pid,
        model,
        model_identity,
    )
    return {
        "schema_version": "fleet-runtime-evidence-probe.v1",
        "node_id": socket.gethostname().split(".", 1)[0],
        "platform": witness._platform_name(),  # noqa: SLF001
        "pid": pid,
        "process": process,
        "model_path": str(model),
        "model_file_identity": model_identity,
        "model_process_binding": binding,
        "strong_binding": binding in _STRONG_BINDINGS,
        "mutating": False,
        "admission_authority": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only runtime evidence probe. It does not create admission, "
            "routing authority, a witness, or a projection."
        )
    )
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    args = parser.parse_args()

    try:
        result = probe(args.pid, args.model_path)
    except (OSError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "schema_version": "fleet-runtime-evidence-probe.v1",
                    "status": "unverified",
                    "error": str(exc),
                    "mutating": False,
                    "admission_authority": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2

    result["status"] = "strong" if result["strong_binding"] else "diagnostic"
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
