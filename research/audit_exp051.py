#!/usr/bin/env python3
"""Source-locked audit of EXP-051 archived aggregate files; never reruns models."""
import argparse
import hashlib
import json
import math
from pathlib import Path
ROOT = Path(__file__).resolve().parent
INPUTS = {
    "EXP051-PROTOCOL-2026-10-16.md": "63c3c50afc2f98bb2d5928da180e7dc1f79f52b7672378ec829fb6f7209c4fdf",
    "k2_qwen4b_exp051.py": "fdd90719644770e685aee84e82ea37b4add347723dce360458f6d668eb190977",
    "EXP051-K2-3P7B-RESULTS-2026-10-16.json": "5e451b7da15c25e828dc667cad6cbc9e0ec9fde4524c9d3e44ff2b57f3aff687",
    "EXP051-QWEN4B-RESULTS-2026-10-16.json": "969c0258e18693709e6d49ae7ccbd2df38064562d9f4366c6b727516a9df073b",
}
FIXTURE_SHA = "af188e9d490d420e02cc67d83a3405d37061fd281012f2dfb67efb8864740f9c"
ARMS = ("supported_original", "supported_swapped", "supported_withheld",
        "unrelated_original", "unrelated_swapped")
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def check_files(root=ROOT):
    for name,expected in INPUTS.items():
        p=root/name
        if not p.is_file() or sha(p)!=expected:
            raise ValueError("HOLD missing/changed source: "+name)
def validate_result(result,model):
    if result["model"]!=model or result["forwards"]!=80:
        raise ValueError("HOLD model/forward mismatch")
    if result["protocol_sha256"]!=INPUTS["EXP051-PROTOCOL-2026-10-16.md"]:
        raise ValueError("HOLD protocol mismatch")
    if result["fixture_sha256"]!=FIXTURE_SHA:
        raise ValueError("HOLD fixture mismatch")
    if not math.isfinite(result["seconds"]) or result["seconds"]<=0:
        raise ValueError("HOLD invalid time")
    summary=result["summary"]
    if set(summary["strict"])!=set(ARMS):
        raise ValueError("HOLD missing design arm")
    if any(summary["strict"][a]!={"correct":16,"n":16} for a in ARMS):
        raise ValueError("HOLD strict cell mismatch")
    for field in ("source_switch","question_x_evidence"):
        stat=summary[field]
        if stat["n_topics"]!=8 or stat["positive_topics"]!=8:
            raise ValueError("HOLD pair/source count mismatch")
        if not math.isfinite(stat["mean"]) or stat["mean"]<=0:
            raise ValueError("HOLD invalid mean")
    if set(result["sources_sha256"])!={"config.json","model.safetensors.index.json","tokenizer.json"}:
        raise ValueError("HOLD custody keys")
    if not all(len(x)==64 for x in result["sources_sha256"].values()):
        raise ValueError("HOLD digest invalid")
    return summary
def audit(root=ROOT):
    check_files(root)
    outputs={}
    for model,filename in (("k2","EXP051-K2-3P7B-RESULTS-2026-10-16.json"),
                           ("qwen","EXP051-QWEN4B-RESULTS-2026-10-16.json")):
        result=json.loads((root/filename).read_text())
        validate_result(result,model)
        outputs[model]=result
    return outputs
def report(data):
    k=data["k2"];q=data["qwen"]
    lines=["# EXP-051 — K2-Horizon-3.7B versus Qwen3.5-4B-Base",
    "", "**2026-10-16 | 160 real BF16 CPU next-token model forwards, eight authored topics.**",
    "", "Frozen protocol SHA256: `"+INPUTS["EXP051-PROTOCOL-2026-10-16.md"]+"`.",
    "Model runner SHA256: `"+INPUTS["k2_qwen4b_exp051.py"]+"`.",
    "Fixture receipt SHA256: `"+FIXTURE_SHA+"`.",
    "", "| Measurement | K2 3.7B | Qwen3.5 4B Base |",
    "|---|---:|---:|"]
    for a in ARMS:
        label=a.replace("_"," ")
        lines.append(f"| {label} strict | 16/16 | 16/16 |")
    lines += [
      "| Total strict | 80/80 | 80/80 |",
      "| Positive evidence-switch topics | 8/8 | 8/8 |",
      "| Positive question × evidence topics | 8/8 | 8/8 |",
      f"| Mean evidence-switch margin | {k['summary']['source_switch']['mean']:.6f} | {q['summary']['source_switch']['mean']:.6f} |",
      f"| Mean question × evidence interaction | {k['summary']['question_x_evidence']['mean']:.6f} | {q['summary']['question_x_evidence']['mean']:.6f} |",
      f"| Native CPU inference seconds | {k['seconds']:.3f} | {q['seconds']:.3f} |",
      "", "## Frozen report digests", ""]
    for f in ("EXP051-K2-3P7B-RESULTS-2026-10-16.json","EXP051-QWEN4B-RESULTS-2026-10-16.json"):
        lines.append("- `"+f+"`: `"+INPUTS[f]+"`.")
    lines += ["", "## Interpretation",
      "- This is a **saturated synthetic QA pilot**: 8 distinct authored topics x 10 conditions each, not 80 independent facts.",
      "- A/B label rotations and swapped/withheld evidence were held constant across models,",
      "  but both succeeded; this task cannot rank semantic grounding or generalization.",
      "- Mean margins are **within-tokenizer descriptors only**, not directly calibrated",
      "  cross-family superiority tests. CPU timings are not controlled throughput benchmarks.",
      "- Qwen is a pretrained Base checkpoint, K2 is separately developed, and vocabulary and",
      "  architecture differ. No direct teacher lineage, hidden-state identity or neuron alignment.",
      "- Different EXP-049/050 smaller-model QASPER results must not be conflated with these.",
      "- Follow with human-adjudicated answer/foil pairs, unseen source-disjoint questions,",
      "  free generation, citation attribution, hard negatives and paired bootstrap uncertainty.",
      "- Draft research only. No weights changed, trained, merged, promoted, routed or deployed;",
      "  no production NAS/Neo4j operations or arXiv submission.",""]
    return "\n".join(lines)
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--out",type=Path)
    args=ap.parse_args()
    results=audit()
    print("EXP051_AUDIT_OK models=2 forwards=160 distinct_topics=8")
    if args.out:
        if args.out.exists() or args.out.parent.resolve()!=ROOT.resolve():
            raise ValueError("HOLD output must be new file in research directory")
        args.out.open("x").write(report(results))
        print("REPORT_WRITTEN",args.out)
if __name__=="__main__":main()