"""Operational hardening shared by the fleet rollout and operator surfaces."""
from __future__ import annotations

from typing import Any, List, Mapping, Sequence, Tuple

SAFE_SSH_OPTIONS: Tuple[str, ...] = (
    "BatchMode=yes",
    "ConnectTimeout=10",
    "ConnectionAttempts=1",
    "ServerAliveInterval=15",
    "ServerAliveCountMax=2",
    "StrictHostKeyChecking=yes",
    "LogLevel=ERROR",
)


def _option_values(argv: Sequence[str], name: str) -> List[str]:
    values = list(argv)
    found: List[str] = []
    index = 0
    while index < len(values):
        value = values[index]
        if value == name:
            if index + 1 >= len(values):
                raise ValueError(f"{name} requires a value")
            found.append(values[index + 1])
            index += 2
            continue
        if value.startswith(name + "="):
            found.append(value.split("=", 1)[1])
        index += 1
    return found


def harden_ssh_argv(argv: Sequence[str]) -> Tuple[List[str], str]:
    """Enforce non-interactive SSH and explicit host-key trust."""
    cleaned: List[str] = []
    allow_accept_new = False
    for value in argv:
        if value == "--allow-accept-new-host-keys":
            allow_accept_new = True
        else:
            cleaned.append(value)

    if not cleaned or cleaned[0] != "run":
        return cleaned, "not_applicable"

    supplied = _option_values(cleaned, "--ssh-option")
    lowered = [value.strip().lower() for value in supplied]
    for value in lowered:
        if value in {
            "stricthostkeychecking=no",
            "stricthostkeychecking=off",
            "stricthostkeychecking=false",
        }:
            raise ValueError("SSH host-key verification may not be disabled")
        if value in {
            "userknownhostsfile=/dev/null",
            "globalknownhostsfile=/dev/null",
        }:
            raise ValueError("SSH known-host verification may not use /dev/null")

    strict_values = [
        value.split("=", 1)[1]
        for value in lowered
        if value.startswith("stricthostkeychecking=")
    ]
    if len(set(strict_values)) > 1:
        raise ValueError("conflicting StrictHostKeyChecking options")
    strict = strict_values[-1] if strict_values else "yes"
    if strict == "accept-new":
        if not allow_accept_new:
            raise ValueError(
                "StrictHostKeyChecking=accept-new requires "
                "--allow-accept-new-host-keys"
            )
        trust_mode = "accept_new_explicit"
    elif strict == "yes":
        trust_mode = "strict_known_hosts"
    else:
        raise ValueError(f"unsupported StrictHostKeyChecking mode: {strict}")

    required = list(SAFE_SSH_OPTIONS)
    if trust_mode == "accept_new_explicit":
        required = [
            "StrictHostKeyChecking=accept-new"
            if value.startswith("StrictHostKeyChecking=")
            else value
            for value in required
        ]
    existing_keys = {
        value.split("=", 1)[0].strip().lower()
        for value in supplied
        if "=" in value
    }
    for option in required:
        key = option.split("=", 1)[0].lower()
        if key not in existing_keys:
            cleaned.extend(["--ssh-option", option])
    return cleaned, trust_mode


def harden_exact_update_script(
    script: str,
    node: Mapping[str, Any],
    update_code: bool,
) -> str:
    """Replace moving-branch pull behavior with an exact-commit fast-forward.

    The configured branch must resolve to the configured commit at fetch time.
    The local branch may advance only when its current commit is an ancestor of
    that exact fetched commit. Divergence, a branch move, or rollback fails
    before benchmark execution.
    """
    if not update_code:
        return script
    expected_commit = str(node.get("expected_commit") or "").lower()
    original = '''git -C "$REPO_DIR" fetch --prune origin "$EXPECTED_BRANCH"
git -C "$REPO_DIR" checkout "$EXPECTED_BRANCH"
git -C "$REPO_DIR" pull --ff-only origin "$EXPECTED_BRANCH"'''
    replacement = f'''test -z "$(git -C "$REPO_DIR" status --porcelain --untracked-files=all)" || {{
  echo "remote checkout is not completely clean before update" >&2
  exit 21
}}
git -C "$REPO_DIR" fetch --prune origin "$EXPECTED_BRANCH"
FETCHED_COMMIT=$(git -C "$REPO_DIR" rev-parse FETCH_HEAD)
test "$FETCHED_COMMIT" = "{expected_commit}" || {{
  echo "origin branch moved: expected {expected_commit}, fetched $FETCHED_COMMIT" >&2
  exit 21
}}
git -C "$REPO_DIR" checkout "$EXPECTED_BRANCH"
CURRENT_COMMIT=$(git -C "$REPO_DIR" rev-parse HEAD)
if [ "$CURRENT_COMMIT" != "$FETCHED_COMMIT" ]; then
  git -C "$REPO_DIR" merge-base --is-ancestor "$CURRENT_COMMIT" "$FETCHED_COMMIT" || {{
    echo "local branch is not a fast-forward ancestor of expected commit" >&2
    exit 21
  }}
  git -C "$REPO_DIR" merge --ff-only "$FETCHED_COMMIT"
fi
UPDATED_COMMIT=$(git -C "$REPO_DIR" rev-parse HEAD)
test "$UPDATED_COMMIT" = "{expected_commit}" || {{
  echo "exact commit update failed: found $UPDATED_COMMIT" >&2
  exit 21
}}'''
    if original not in script:
        raise RuntimeError("rollout update block changed unexpectedly")
    return script.replace(original, replacement, 1)


def remote_lock_and_provenance_snippet(
    base: Any,
    node: Mapping[str, Any],
    run_id: str,
) -> str:
    """Return portable mkdir-lock logic with safe stale-lock recovery."""
    lock_root = str(node.get("lock_root") or "~/.local/state/lms-fleet/locks")
    node_slug = base.safe_slug(str(node["node_id"]))
    expected_commit = str(node.get("expected_commit") or "").lower()
    return f'''RAW_LOCK_ROOT={base.q(lock_root)}
LOCK_ROOT=$($PYTHON_BIN - "$RAW_LOCK_ROOT" <<'PY'
import os, sys
print(os.path.abspath(os.path.expanduser(sys.argv[1])))
PY
)
mkdir -p "$LOCK_ROOT"
LOCK_DIR="$LOCK_ROOT/{node_slug}.lock"
LOCK_ACQUIRED=0
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  LOCK_STATE=$($PYTHON_BIN - "$LOCK_DIR/owner.json" <<'PY'
import json, os, pathlib, socket, sys
path = pathlib.Path(sys.argv[1])
try:
    owner = json.loads(path.read_text(encoding='utf-8'))
except (OSError, json.JSONDecodeError):
    print('unknown')
    raise SystemExit(0)
if owner.get('hostname') != socket.gethostname():
    print('foreign')
    raise SystemExit(0)
def boot_id():
    try:
        return pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    except OSError:
        return None
current_boot = boot_id()
owner_boot = owner.get('boot_id')
if current_boot and owner_boot and current_boot != owner_boot:
    print('stale')
    raise SystemExit(0)
try:
    pid = int(owner.get('pid'))
    os.kill(pid, 0)
except (TypeError, ValueError, ProcessLookupError):
    print('stale')
except PermissionError:
    print('active')
else:
    print('active')
PY
)
  if [ "$LOCK_STATE" = "stale" ]; then
    STALE_LOCK="$LOCK_DIR.stale.$(date -u +%Y%m%dT%H%M%SZ).$$"
    mv -- "$LOCK_DIR" "$STALE_LOCK" || {{
      echo "failed to archive stale LMS fleet lock $LOCK_DIR" >&2
      exit 22
    }}
    mkdir "$LOCK_DIR" || {{
      echo "failed to acquire LMS fleet lock after stale recovery" >&2
      exit 22
    }}
    echo "archived stale LMS fleet lock as $STALE_LOCK" >&2
  else
    echo "another LMS fleet rollout holds or ambiguously owns $LOCK_DIR" >&2
    if [ -f "$LOCK_DIR/owner.json" ]; then
      cat "$LOCK_DIR/owner.json" >&2
    fi
    exit 22
  fi
fi
LOCK_ACQUIRED=1
SHELL_OWNER_PID=${{BASHPID:-$$}}
"$PYTHON_BIN" - "$LOCK_DIR/owner.json" "$SHELL_OWNER_PID" <<'PY'
import datetime as dt, json, os, pathlib, socket, sys
path = pathlib.Path(sys.argv[1])
pid = int(sys.argv[2])
try:
    boot = pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()
except OSError:
    boot = None
payload = {{
    'schema_version': 'fleet_remote_lock.v2',
    'node_id': os.environ.get('NODE_ID'),
    'run_id': os.environ.get('RUN_ID'),
    'hostname': socket.gethostname(),
    'pid': pid,
    'boot_id': boot,
    'started_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
}}
temporary = path.with_suffix('.tmp')
with temporary.open('w', encoding='utf-8') as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write('\\n')
    handle.flush()
    os.fsync(handle.fileno())
os.replace(temporary, path)
PY
"$PYTHON_BIN" -m lms_agent_bench.fleet_provenance \\
  --repo "$REPO_DIR" \\
  --node-id "$NODE_ID" \\
  --run-id "$RUN_ID" \\
  --expected-branch "$EXPECTED_BRANCH" \\
  --expected-commit {base.q(expected_commit)} \\
  --out "$ARTIFACT_DIR/source_control.json"
'''


def apply_entrypoint_hardening(entrypoint: Any) -> None:
    entrypoint._lock_and_provenance_snippet = (  # noqa: SLF001
        lambda node, run_id: remote_lock_and_provenance_snippet(
            entrypoint._base, node, run_id  # noqa: SLF001
        )
    )
    if not getattr(entrypoint, "_lms_exact_update_hardened", False):
        original_builder = entrypoint._ORIGINAL_BUILD_REMOTE_SCRIPT  # noqa: SLF001

        def exact_builder(
            node: Mapping[str, Any],
            run_id: str,
            execute_candidates: Sequence[str] = (),
            update_code: bool = False,
            dry_run_limit: int = 4,
        ) -> str:
            generated = original_builder(
                node,
                run_id,
                execute_candidates=execute_candidates,
                update_code=update_code,
                dry_run_limit=dry_run_limit,
            )
            return harden_exact_update_script(generated, node, update_code)

        entrypoint._ORIGINAL_BUILD_REMOTE_SCRIPT = exact_builder  # noqa: SLF001
        entrypoint._lms_exact_update_hardened = True  # noqa: SLF001
