"""Isolated entrypoint for one strict benchmark measurement trial."""
from __future__ import annotations

import sys

from lms_agent_bench import lms_eval as _lms_eval

sys.modules["lms_eval"] = _lms_eval

from lms_agent_bench import benchmark_lmstudio_cross_machine_models as _legacy
from lms_agent_bench import benchmark_protocol as _protocol

_legacy.call_chat_completions_stream = _protocol.call_chat_completions_stream
_legacy.call_chat_completions_once = _protocol.call_chat_completions_once
main = _legacy.main

__all__ = ["main"]

if __name__ == "__main__":
    raise SystemExit(main())
