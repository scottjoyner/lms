#!/usr/bin/env python3
"""Create redacted LMS support bundles.

The support bundle is designed for debugging CI/user issues without leaking raw
model outputs, API keys, bearer tokens, private keys, or arbitrary local files.
Raw model outputs are excluded by default and require --include-raw-outputs.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


DEFAULT_INCLUDE_FILES = [
    "lms_run_config.json",
    "endpoint_probes.json",
    "machine_profile.json",
    "machine_synopsis.md",
    "lmstudio_inventory.csv",
    "config.json",
    "run_results.csv",
    "run_summary.csv",
    "task_summary.csv",
    "capability_matrix.csv",
    "agent_recommendations.md",
    "routing_rules.yaml",
    "routing_rules.json",
    "model_fit.csv",
    "model_fit.md",
    "AGENT_BRIEF.md",
    "run_audit.json",
    "RUN_AUDIT.md",
    "lms_agent_skill.json",
    "LMS_AGENT_SKILL.md",
    "artifact_validation.json",
    "ARTIFACT_VALIDATION.md",
]

RAW_OUTPUT_MARKERS = ["/outputs/", "sidecars/run_"]

SECRET_PATTERNS = [
    (re.compile(r"AKIA[0-9A-Z]{16}"), "[REDACTED_AWS_ACCESS_KEY]"),
    (re.compile(r"ASIA[0-9A-Z]{16}"), "[REDACTED_AWS_TEMP_KEY]"),
    (re.compile(r"sk-[A-Za-z0-9_-]{20,}"), "[REDACTED_OPENAI_STYLE_KEY]"),
    (re.compile(r"ghp_[A-Za-z0-9]{20,}"), "[REDACTED_GITHUB_TOKEN]"),
    (re.compile(r"github_pat_[A-Za-z0-9_]{20,}"), "[REDACTED_GITHUB_PAT]"),
    (re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"), "[REDACTED_SLACK_TOKEN]"),
    (re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----.*?-----END (?:RSA |OPENSSH |EC )?PRIVATE KEY-----", re.DOTALL), "[REDACTED_PRIVATE_KEY]"),
    (re.compile(r"(?i)(authorization\s*[:=]\s*bearer\s+)[A-Za-z0-9._~+/=-]{16,}"), r"\1[REDACTED_BEARER_TOKEN]"),
    (re.compile(r"(?i)((api[_-]?key|secret|token|password)\s*[:=]\s*['\"]?)[A-Za-z0-9_./+=-]{12,}"), r"\1[REDACTED_SECRET]"),
]


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "run"


def redact_text(text: str) -> Tuple[str, int]:
    count = 0
    for pattern, repl in SECRET_PATTERNS:
        text, n = pattern.subn(repl, text)
        count += n
    return text, count


def should_include(path: Path, run_dir: Path, include_raw_outputs: bool) -> bool:
    rel = str(path.relative_to(run_dir)).replace("\\", "/")
    if path.name in DEFAULT_INCLUDE_FILES:
        return True
    if rel.startswith("sidecars/") and path.suffix.lower() in {".md", ".json", ".csv"}:
        if not include_raw_outputs and "/outputs/" in rel:
            return False
        return True
    if include_raw_outputs and rel.startswith("sidecars/") and path.suffix.lower() in {".txt", ".md", ".json", ".csv"}:
        return True
    return False


def collect_files(run_dir: Path, include_raw_outputs: bool) -> List[Path]:
    files: List[Path] = []
    for path in run_dir.rglob("*"):
        if path.is_file() and should_include(path, run_dir, include_raw_outputs):
            files.append(path)
    return sorted(set(files))


def read_redacted(path: Path) -> Tuple[bytes, int, bool]:
    try:
        raw = path.read_bytes()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            return raw, 0, False
        redacted, count = redact_text(text)
        return redacted.encode("utf-8"), count, True
    except Exception as exc:
        return f"[ERROR_READING_FILE] {exc}\n".encode("utf-8"), 0, True


def create_bundle(run_dir: Path, out: Optional[Path] = None, include_raw_outputs: bool = False) -> Dict[str, Any]:
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)
    out = out or (run_dir / f"lms_support_bundle_{safe_name(run_dir.name)}.zip")
    out.parent.mkdir(parents=True, exist_ok=True)
    files = collect_files(run_dir, include_raw_outputs=include_raw_outputs)
    manifest: Dict[str, Any] = {
        "schema_version": "lms_support_bundle.v1",
        "generated_at_utc": utc_now_iso(),
        "run_dir": str(run_dir),
        "include_raw_outputs": include_raw_outputs,
        "bundle_path": str(out),
        "files": [],
        "redaction_count": 0,
    }
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            rel = path.relative_to(run_dir)
            data, redactions, text_mode = read_redacted(path)
            manifest["redaction_count"] += redactions
            manifest["files"].append({
                "path": str(rel).replace("\\", "/"),
                "bytes": len(data),
                "text_mode": text_mode,
                "redactions": redactions,
            })
            zf.writestr(str(rel).replace("\\", "/"), data)
        zf.writestr("SUPPORT_BUNDLE_MANIFEST.json", json.dumps(manifest, indent=2, sort_keys=True))
        zf.writestr("README_SUPPORT_BUNDLE.md", render_readme(manifest))
    return manifest


def render_readme(manifest: Dict[str, Any]) -> str:
    lines = [
        "# LMS Support Bundle",
        "",
        f"- Generated UTC: `{manifest.get('generated_at_utc')}`",
        f"- Source run: `{manifest.get('run_dir')}`",
        f"- Raw outputs included: `{manifest.get('include_raw_outputs')}`",
        f"- Redactions applied: `{manifest.get('redaction_count')}`",
        "",
        "## Privacy notes",
        "",
        "- Secrets are regex-redacted on a best-effort basis.",
        "- Raw model output files are excluded unless `--include-raw-outputs` was used.",
        "- Review this zip before sharing outside your environment.",
        "",
        "## Included files",
        "",
    ]
    for item in manifest.get("files", []):
        lines.append(f"- `{item['path']}` ({item['bytes']} bytes, redactions={item['redactions']})")
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a redacted support bundle for an LMS run directory.")
    parser.add_argument("run_dir")
    parser.add_argument("--out", default=None)
    parser.add_argument("--include-raw-outputs", action="store_true")
    parser.add_argument("--pretty", action="store_true")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = create_bundle(Path(args.run_dir), out=Path(args.out) if args.out else None, include_raw_outputs=args.include_raw_outputs)
    print(json.dumps(manifest, indent=2 if args.pretty else None, sort_keys=True))
    print(f"wrote {manifest['bundle_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
