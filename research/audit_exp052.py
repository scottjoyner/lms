#!/usr/bin/env python3
"""Immutable three-checkpoint EXP051+EXP052 aggregate audit; no model inference."""
import argparse,json,math
from pathlib import Path
from audit_exp051 import audit as audit051, ARMS, ROOT, sha
EXPECTED={
"EXP052-PROTOCOL-2026-10-16.md":"d9d95a879f3c12a5168c86e7f24969e2e6ce92a6adcfc55b92b7cb4d6f651bee",
"minicpm_v47_exp052.py":"e83093ab5c28527cbb5be6e5f71397898a60df8371155c0013d3f38032e7953c",
"EXP052-MINICPM-RESULTS-2026-10-16.json":"d9c69b06925e5f2b7beb5f93d6f1a49b12e84c219010d95aca9999f23fe2d475",
}
FIXTURE="af188e9d490d420e02cc67d83a3405d37061fd281012f2dfb67efb8864740f9c"
RUNNER051="fdd90719644770e685aee84e82ea37b4add347723dce360458f6d668eb190977"
REV="286817f1132b998913a129a221c89b4402b4ab80"
def verify_mini(data):
    if (data["model"],data["hf_revision"],data["forward_passes"])!=("minicpm-v4.7-1b",REV,80):
        raise ValueError("HOLD MiniCPM model/revision/forwards")
    if data["protocol_sha256"]!=EXPECTED["EXP052-PROTOCOL-2026-10-16.md"]:
        raise ValueError("HOLD protocol mismatch")
    if data["exp051_fixture_sha256"]!=FIXTURE or data["exp051_runner_sha256"]!=RUNNER051:
        raise ValueError("HOLD EXP051 inherited fixture")
    if data["transformers_version"]!="5.20.1":raise ValueError("HOLD runtime")
    if set(data["sources_sha256"])!={"config.json","tokenizer.json","tokenizer_config.json","model.safetensors"}:
        raise ValueError("HOLD checkpoint receipt keys")
    if any(len(s)!=64 for s in data["sources_sha256"].values()):
        raise ValueError("HOLD digest lengths")
    if not math.isfinite(data["timed_forward_seconds"]) or data["timed_forward_seconds"]<=0:
        raise ValueError("HOLD inference duration")
    summ=data["summary"]
    if set(summ["strict"])!=set(ARMS):raise ValueError("HOLD missing task arms")
    if any(summ["strict"][a]!={"correct":16,"n":16} for a in ARMS):
        raise ValueError("HOLD strict count changed")
    for s in ("source_switch","question_x_evidence"):
        item=summ[s]
        if item["n_topics"]!=8 or item["positive_topics"]!=8:
            raise ValueError("HOLD topic sign count")
        if not math.isfinite(item["mean"]) or item["mean"]<=0:
            raise ValueError("HOLD sign/margin")
    return summ
def audit():
    baseline=audit051()
    for name,expected in EXPECTED.items():
        if not (ROOT/name).is_file() or sha(ROOT/name)!=expected:
            raise ValueError("HOLD source change "+name)
    minicpm=json.loads((ROOT/"EXP052-MINICPM-RESULTS-2026-10-16.json").read_text())
    verify_mini(minicpm)
    return baseline,minicpm
def report(baseline,mini):
    data=[("K2-Horizon-3.7B",baseline["k2"]["summary"],baseline["k2"]["seconds"]),
          ("Qwen3.5-4B-Base",baseline["qwen"]["summary"],baseline["qwen"]["seconds"]),
          ("MiniCPM-V-4.7-1B",mini["summary"],mini["timed_forward_seconds"])]
    lines=["# EXP-052 — MiniCPM-V 4.7-1B extension to EXP-051",
    "", "**2026-10-16 | 240 real BF16 CPU text-only forward passes: 80/model.**",
    "", "## Source custody", "",
    "- EXP051 frozen audit and report digests remain unchanged.",
    "- EXP052 frozen protocol SHA256: `"+EXPECTED["EXP052-PROTOCOL-2026-10-16.md"]+"`.",
    "- EXP052 executable SHA256: `"+EXPECTED["minicpm_v47_exp052.py"]+"`.",
    "- EXP052 result SHA256: `"+EXPECTED["EXP052-MINICPM-RESULTS-2026-10-16.json"]+"`.",
    "- MiniCPM-V 4.7-1B HF revision: `"+REV+"` (full BF16 checkpoint pinned).",
    "", "## Measured descriptive results", "",
    "| Measurement | K2 3.7B | Qwen3.5 4B Base | MiniCPM-V 4.7 1B |",
    "|---|---:|---:|---:|"]
    for arm in ARMS:
        vals=[f"{x[1]['strict'][arm]['correct']}/16" for x in data]
        lines.append("| "+arm.replace("_"," ")+" | "+" | ".join(vals)+" |")
    lines+=["| Total strict | 80/80 | 80/80 | 80/80 |",
        "| Positive question × evidence | 8/8 | 8/8 | 8/8 |",
        "| Mean within-model question × evidence effect (nats) | "+
        " | ".join(f"{x[1]['question_x_evidence']['mean']:.6f}" for x in data)+" |",
        "| CPU native forward time (seconds) | "+
        " | ".join(f"{x[2]:.3f}" for x in data)+" |"]
    lines+=["", "## Required limits / next falsification", "",
    "- All three models **saturate** this eight-topic authored synthetic test. No relative QA ability,",
    "  human-independent semantic understanding, causal neuron or K2/Qwen teacher lineage follows.",
    "- `NOT STATED` is a fixed provided option; no free response or citation attribution.",
    "- MiniCPM is instruction-tuned/multimodal, yet this study uses raw text without its native",
    "  chat template, no image frames. K2/Qwen Base have different size and training.",
    "- Raw log-prob margins are not calibrated across tokenizers or runtimes. Seconds do not",
    "  establish benchmark throughput; kernels, CPU load and model sizes differ.",
    "- Stronger next cohort: independent human-adjudicated paper-disjoint supported and plausible",
    "  unsupported answer alternatives, controlled foil length, swapped passage, wrong question,",
    "  generation and abstention with paired article-level uncertainty.",
    "- Draft-only: no weights, merging, live routing, production graph/NAS writes, or arXiv.",""]
    return "\n".join(lines)
def main():
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path)
    args=p.parse_args()
    b,m=audit()
    print("EXP052_AUDIT_OK models=3 forwards=240 unique_topics=8")
    if args.out:
        if args.out.exists() or args.out.parent.resolve()!=ROOT.resolve():
            raise ValueError("HOLD unique local research report required")
        args.out.open("x").write(report(b,m))
        print("REPORT_WRITTEN",args.out)
if __name__=="__main__":main()