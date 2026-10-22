#!/usr/bin/env bash
set -euo pipefail
ROOT=${1:-$PWD}
RESULT_ROOT=${RESULT_ROOT:-$ROOT/results}
LLAMA_BUILD=${LLAMA_BUILD:-$HOME/src/llama.cpp/build-rocm-maxout}

status=NOT_STARTED
reason="no physical evidence found"

preflight=$(find "$RESULT_ROOT" -maxdepth 2 -type f \( -name 'rocminfo.txt' -o -name 'storage.txt' -o -name 'host.txt' \) 2>/dev/null | head -1 || true)
[[ -n "$preflight" ]] && { status=PREFLIGHT_DONE; reason="preflight evidence exists"; }

if [[ -x "$LLAMA_BUILD/bin/llama-server" && -x "$LLAMA_BUILD/bin/llama-bench" ]]; then
  status=BUILD_DONE
  reason="experiment llama.cpp build exists"
fi

started=$(find "$RESULT_ROOT" -type f \( -name '*.jsonl' -o -name '*-warm.json' -o -name '*-cold.json' -o -name '*.prom' \) 2>/dev/null | head -1 || true)
[[ -n "$started" ]] && { status=BENCHMARK_STARTED; reason="benchmark samples exist"; }

complete=$(find "$RESULT_ROOT" -type f -name RESULT_DIR.txt 2>/dev/null | head -1 || true)
[[ -n "$complete" ]] && { status=BENCHMARK_COMPLETE; reason="runner completion marker exists"; }

python3 - "$status" "$reason" "$ROOT" "$RESULT_ROOT" "$LLAMA_BUILD" <<'PY'
import json,os,subprocess,sys,datetime
status,reason,root,result_root,llama_build=sys.argv[1:]
def cmd(c):
    try:return subprocess.check_output(c,shell=True,text=True,stderr=subprocess.STDOUT).strip()
    except Exception:return None
obj={
 "schema":"lms.rocm_execution_checkpoint.v1",
 "timestamp_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
 "hostname":os.uname().nodename,
 "status":status,
 "reason":reason,
 "repo_head":cmd(f"git -C {root!r} rev-parse HEAD"),
 "repo_branch":cmd(f"git -C {root!r} branch --show-current"),
 "working_tree_clean":cmd(f"git -C {root!r} status --porcelain") == "",
 "llama_server_version":cmd(f"{llama_build!r}/bin/llama-server --version"),
 "result_root":result_root,
 "result_file_count":int(cmd(f"find {result_root!r} -type f 2>/dev/null | wc -l") or 0),
 "completion_markers":int(cmd(f"find {result_root!r} -type f -name RESULT_DIR.txt 2>/dev/null | wc -l") or 0),
}
print(json.dumps(obj,indent=2,sort_keys=True))
PY
