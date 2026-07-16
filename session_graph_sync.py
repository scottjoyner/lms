# Backwards-compatible shim. The canonical implementation now lives in the
# lms_agent_bench package (src/lms_agent_bench/session_graph_sync.py). Importing this module
# re-exports it so legacy callers keep working during the packaging migration.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from lms_agent_bench.session_graph_sync import *  # noqa: F401,F403

if __name__ == "__main__":
    raise SystemExit(main())
