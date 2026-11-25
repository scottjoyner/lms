"""Package-level CLI entrypoint.

Exposes :func:`main` (delegating to the canonical ``lms_cli.main``) so the
``lms`` console script can point at ``lms_agent_bench.cli:main`` while the
README's ``lms = "lms_cli:main"`` contract still holds (``lms_cli`` keeps its
own ``main()``). See docs/LLD_UNIFIED_FLEET.md §3.6 (W-67).
"""

from __future__ import annotations

from lms_agent_bench.lms_cli import main as main

__all__ = ["main"]
