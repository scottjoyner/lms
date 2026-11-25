#!/usr/bin/env python3
"""Estimate model-to-hardware fit for LMS runs.

This is a heuristic helper. LM Studio does not always expose full model metadata,
so this parser estimates parameter class, quantization, and likely memory needs
from model names. The output is intended as a routing warning, not a guarantee.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


QUANT_BITS = {
    "q2": 2.5,
    "q3": 3.5,
    "q4": 4.5,
    "q5": 5.5,
    "q6": 6.5,
    "q8": 8.5,
    "int4": 4.5,
    "int8": 8.5,
    "fp16": 16.0,
    "f16": 16.0,
    "bf16": 16.0,
    "fp32": 32.0,
    "f32": 32.0,
}


def read_json(path: Path) -> Dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, Any]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{k: row.get(k, "") for k in fields} for row in rows])


def gib(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return round(float(value) / (1024 ** 3), 2)
    except (TypeError, ValueError):
        return None


def parse_params_b(model_name: str) -> Optional[float]:
    text = model_name.lower()
    patterns = [
        r"(?:^|[-_/ .])([0-9]+(?:\.[0-9]+)?)\s*b(?:[-_/ .]|$)",
        r"(?:^|[-_/ .])([0-9]+(?:\.[0-9]+)?)\s*bn(?:[-_/ .]|$)",
        r"(?:^|[-_/ .])([0-9]+(?:\.[0-9]+)?)\s*billion(?:[-_/ .]|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return float(match.group(1))
    return None


def parse_quant(model_name: str) -> Tuple[str, float]:
    text = model_name.lower()
    # Prefer concrete GGUF-style quants such as Q4_K_M.
    match = re.search(r"\b(q[2-8](?:_[a-z0-9]+)*)\b", text)
    if match:
        quant = match.group(1)
        family = quant[:2]
        return quant.upper(), QUANT_BITS.get(family, 4.5)
    for key, bits in QUANT_BITS.items():
        if re.search(rf"\b{re.escape(key)}\b", text):
            return key.upper(), bits
    # Many LM Studio model names omit quantization. Assume Q4-ish for local GGUF,
    # but keep that clear in notes.
    return "unknown_assume_q4", 4.5


def estimate_memory_gib(params_b: Optional[float], bits_per_param: float) -> Optional[float]:
    if params_b is None:
        return None
    # params_b billion params * bits / 8 = GB decimal. Convert-ish to GiB and
    # add KV/cache/runtime overhead factor. This is heuristic by design.
    raw_gb = params_b * bits_per_param / 8.0
    overhead = max(1.2, 1.05 + min(params_b / 100.0, 0.35))
    return round(raw_gb * overhead * 0.931, 2)


def extract_nvidia_vram_gib(profile: Dict[str, Any]) -> Optional[float]:
    devices = (((profile.get("gpu") or {}).get("nvidia") or {}).get("devices") or [])
    totals = []
    for dev in devices:
        raw = dev.get("memory.total")
        try:
            totals.append(float(raw) / 1024.0)
        except (TypeError, ValueError):
            pass
    return round(max(totals), 2) if totals else None


def machine_memory(profile: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    memory = profile.get("memory") or {}
    ram = gib(memory.get("mem_total_bytes"))
    available = gib(memory.get("mem_available_bytes"))
    return ram, available


def fit_grade(estimated_gib: Optional[float], ram_gib: Optional[float], available_gib: Optional[float], vram_gib: Optional[float]) -> Tuple[str, str]:
    if estimated_gib is None:
        return "unknown", "Could not estimate model size from name. Run benchmark load checks and monitor LM Studio memory."
    best_memory = max([x for x in [available_gib, vram_gib] if x is not None] or [0])
    total_memory = max([x for x in [ram_gib, vram_gib] if x is not None] or [0])
    if best_memory and estimated_gib <= best_memory * 0.75:
        return "good", "Estimated model memory fits comfortably in currently available RAM/VRAM."
    if total_memory and estimated_gib <= total_memory * 0.85:
        return "borderline", "Estimated model may fit, but expect pressure from KV cache, context length, and other processes."
    if total_memory and estimated_gib <= total_memory * 1.15:
        return "risky", "Estimated model is near machine memory limits; use small context and expect load failures or swapping."
    return "poor", "Estimated model exceeds comfortable local memory limits for this host."


def analyze_model(model_key: str, profile: Dict[str, Any]) -> Dict[str, Any]:
    params_b = parse_params_b(model_key)
    quant, bits = parse_quant(model_key)
    estimated = estimate_memory_gib(params_b, bits)
    ram, available = machine_memory(profile)
    vram = extract_nvidia_vram_gib(profile)
    fit, note = fit_grade(estimated, ram, available, vram)
    return {
        "model_key": model_key,
        "estimated_params_b": params_b if params_b is not None else "",
        "estimated_quant": quant,
        "estimated_bits_per_param": bits,
        "estimated_model_memory_gib": estimated if estimated is not None else "",
        "system_ram_gib": ram if ram is not None else "",
        "available_ram_gib": available if available is not None else "",
        "largest_nvidia_vram_gib": vram if vram is not None else "",
        "fit_grade": fit,
        "fit_notes": note,
    }


def rows_from_run(run_dir: Path) -> List[str]:
    models = set()
    for path_name in ["capability_matrix.csv", "lmstudio_inventory.csv", "run_summary.csv"]:
        for row in read_csv(run_dir / path_name):
            if row.get("model_key"):
                models.add(row["model_key"])
    return sorted(models)


def analyze_run(run_dir: Path) -> Tuple[List[Dict[str, Any]], str]:
    profile = read_json(run_dir / "machine_profile.json")
    rows = [analyze_model(model, profile) for model in rows_from_run(run_dir)]
    md = render_markdown(run_dir, rows)
    return rows, md


def render_markdown(run_dir: Path, rows: List[Dict[str, Any]]) -> str:
    lines = ["# LMS Model Fit Report", "", f"- Run directory: `{run_dir}`", ""]
    if not rows:
        lines.append("No models found in run artifacts.")
        return "\n".join(lines)
    lines += ["| Model | Params B | Quant | Est. GiB | Fit | Notes |", "|---|---:|---|---:|---|---|"]
    for row in rows:
        lines.append(
            f"| `{row['model_key']}` | {row['estimated_params_b']} | {row['estimated_quant']} | "
            f"{row['estimated_model_memory_gib']} | {row['fit_grade']} | {row['fit_notes']} |"
        )
    lines += ["", "## Notes", "", "- These estimates are heuristic and based on model naming conventions.", "- Actual fit depends on LM Studio backend, KV cache, context length, GPU offload, drivers, and other running processes.", "- Benchmark load success and runtime stability remain the source of truth.", ""]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Estimate model-to-hardware fit for an LMS run directory.")
    parser.add_argument("run_dir", help="LMS run directory")
    parser.add_argument("--csv-out", default=None, help="Optional CSV output path; default run_dir/model_fit.csv")
    parser.add_argument("--md-out", default=None, help="Optional Markdown output path; default run_dir/model_fit.md")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    run_dir = Path(args.run_dir)
    rows, md = analyze_run(run_dir)
    csv_out = Path(args.csv_out) if args.csv_out else run_dir / "model_fit.csv"
    md_out = Path(args.md_out) if args.md_out else run_dir / "model_fit.md"
    fields = ["model_key", "estimated_params_b", "estimated_quant", "estimated_bits_per_param", "estimated_model_memory_gib", "system_ram_gib", "available_ram_gib", "largest_nvidia_vram_gib", "fit_grade", "fit_notes"]
    write_csv(csv_out, rows, fields)
    md_out.write_text(md, encoding="utf-8")
    print(f"wrote {csv_out}")
    print(f"wrote {md_out}")
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
