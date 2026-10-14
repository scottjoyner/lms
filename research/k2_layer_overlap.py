#!/usr/bin/env python3
"""LMS-style reproducible depth CKA and behavioral-lineage comparisons.

Input: pinned historical K2 aggregate evidence. Output: human-readable Markdown.
No new model inference; CKA is not a parameter ancestry or teacher test.
"""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

PINS={
 "018":("results_exp018_intermediate_k2_20261014.json","3818a421a9b9cbe6f52b0a27d8c3da69be0ee1bd4d50071c190d50ba95ee893a"),
 "016":("results_exp016_k2_pretrain_release_20261014.json","8750eadab1bd6a042cf491de87e14aee4eabc98e9eb622dc5169dd0ed4a7bfa5"),
 "019":("results_exp019_fact_bias_20261014.json","12546abb9ed90e4e51630a42415a0cd801bb0642db8a375f706e960ed61fcaf6"),
 "021":("results_exp021_reconciled_20261014.json","3d08b4a90fd2404e0fba0b73f3f18bc9cbd937b8315d80a66d3cb4b88b8f9894"),
}
STAGES=("pretrain600k","mid1_75k","mid2_47684","rl_merged","mopd_249","main")
K2=(0,7,14,21,28)
QWEN=(0,6,12,18,24)

def pinned(root):
    output={}
    for key,(name,expected) in PINS.items():
        raw=(root/name).read_bytes()
        digest=hashlib.sha256(raw).hexdigest()
        if digest!=expected:raise ValueError("SHA256 custody mismatch "+name+" "+digest)
        output[key]=json.loads(raw)
    if tuple(output["018"]["checkpoint_order"])!=STAGES:
        raise ValueError("K2 checkpoint order changed")
    return output

def grid(rows,mode,metric,left=K2,right=K2):
    found={}
    for r in rows:
        k=(r["pooling"],r["left_layer"],r["right_layer"])
        if k in found:raise ValueError("Duplicate layer cell "+str(k))
        found[k]=r
    expected={(p,a,b) for p in ("mean","last") for a in left for b in right}
    if set(found)!=expected:raise ValueError("Missing/extra layer CKA cells")
    m=[]
    for a in left:
        line=[]
        for b in right:
            v=found[mode,a,b][metric]
            if v is not None and (not math.isfinite(v) or v < -1e-6 or v>1.00001):
                raise ValueError("Invalid CKA value")
            line.append(v)
        m.append(line)
    return m

def fmt(v):
    return "—" if v is None else f"{v:.4f}"

def markdown_grid(left,right,m):
    lines=["| Layer \\ layer | "+" | ".join(map(str,right))+" |",
           "|---|"+ "|".join("---:" for _ in right)+"|"]
    lines += [f"| **{layer}** | "+" | ".join(fmt(v) for v in vals)+" |"
              for layer,vals in zip(left,m)]
    return "\n".join(lines)

def depth_diagnostics(m):
    n=len(m)
    if n!=5 or any(len(row)!=5 for row in m):raise ValueError("Need full 5x5")
    diag=[m[i][i] for i in range(n)]
    valid=[x for x in diag if x is not None]
    if not valid:raise ValueError("All diagonal CKA undefined")
    row_match=[max((j for j in range(n) if m[i][j] is not None),
                   key=lambda j:m[i][j],default=None) for i in range(n)]
    col_match=[max((i for i in range(n) if m[i][j] is not None),
                   key=lambda i:m[i][j],default=None) for j in range(n)]
    mutual=[(i,j) for i,j in enumerate(row_match) if j is not None and col_match[j]==i]
    # Undefined CKA cells are not zero. Match only measurable rows and
    # columns; e.g., last-token embedding vectors can be degenerate.
    valid_rows=[i for i in range(n) if row_match[i] is not None]
    valid_cols=[j for j in range(n) if col_match[j] is not None]
    assignments=[]
    if len(valid_rows)<=len(valid_cols):
        for p in itertools.permutations(valid_cols,len(valid_rows)):
            if all(m[i][j] is not None for i,j in zip(valid_rows,p)):
                assignment=[None]*n
                for i,j in zip(valid_rows,p):assignment[i]=j
                assignments.append((sum(m[i][j] for i,j in zip(valid_rows,p))/len(valid_rows),
                                    tuple(assignment)))
    opt=max(assignments) if assignments else None
    return {"diag_mean":sum(valid)/len(valid),
            "final":diag[-1],"row_best":row_match,
            "mutual":mutual,"diagonal_match_count":sum(i==j for i,j in enumerate(row_match)),
            "optimal_mean":opt[0] if opt else None,
            "optimal_assignment":opt[1] if opt else None}

def report(evidence):
    exp=evidence["018"];cross=evidence["016"];behavior=evidence["019"];gen=evidence["021"]
    if (exp.get("experiment"),cross.get("experiment"),behavior.get("experiment"),
        gen.get("experiment"))!=("EXP018","EXP016","EXP019","EXP021-reconciled"):
        raise ValueError("Wrong source schemas")
    lines=["# K2 model lineage: full 5×5 layer-similarity matrices",
      "",
      "**October 14, 2026 — LMS read-only reanalysis of pinned EXP-016/018/019/021.**",
      "This is newly computed layer-alignment *analysis*, not new neural inference.",
      "It uses previously measured CKA cells, and records observations in Markdown.",
      "",
      "## Source custody",
      "",
      "| Experiment | File SHA256 |", "|---|---|"]
    for key,(name,sha) in PINS.items():
        lines.append(f"| {key} — `{name}` | `{sha}` |")
    lines += ["",
      f"Same 64 authored role-reversal statements, {exp['source_groups']} name-pair groups.",
      "K2 sampled depths 0/7/14/21/28; Qwen3.5-0.8B depths 0/6/12/18/24.",
      "Mean pooling and last-nonpadding-token pooling must be read separately.",
      "Signed-role CKA is invariant under D→−D; it cannot establish direction correctness.",
      "",
      "## Behavior lineage, not collapsed with CKA","",
      "| K2 state | A/B correct/256 | Actor/128 | Recipient/128 | A chosen |",
      "|---|---:|---:|---:|---:|"]
    for stage in STAGES:
        item=exp["behavior_by_stage"][stage];all_=item["overall"]
        lines.append(f"| {stage} | {all_['correct']} | "
                     f"{item['role_type']['actor']['correct']} | "
                     f"{item['role_type']['target']['correct']} | "
                     f"{all_['fraction_prefer_A']*100:.1f}% |")
    lines += ["","**Mid2→RL merge:** actor +32/128, recipient −20/128 on this fixture.",
       "A/B preference and actor heuristics are confounded; these are not causal training effects.",
       "",
       "| Stage | EXP019 both option orders/128 | EXP019 debiased/128 | EXP019 positive fact signal/128 | EXP021 strict generated/64 | EXP021 UNKNOWN abstained/64 |",
       "|---|---:|---:|---:|---:|---:|"]
    for stage,k in (("mid2_47684","mid2_47684"),("rl_merged","rl_merged"),
                    ("mopd_249","mopd_249"),("qwen08 (native chat)","qwen08_chat")):
        key=stage.split(" ")[0]
        f=behavior["stages"][key]["overall"];g=gen["summaries"][k]
        lines.append(f"| {stage} | {f['factual_pair_both_correct']} | "
          f"{f['factual_debiased_correct']} | {f['fact_signal_positive']} | "
          f"{g['factual']['strict']} | {g['neutral']['neutral_abstain']} |")
    lines += ["","EXP021 generation is capped at 12 tokens and plain runs truncate heavily.",
       "Native-chat Qwen is not comparable with an invented K2 chat wrapper.",
       "",
       "## Transition synopsis: geometry versus behavior","",
       "| K2 transition | Mean depth diagonal CKA | Last-token depth diagonal CKA | Final-layer mean CKA | Final-layer last CKA | Mutual mean-depth matches |",
       "|---|---:|---:|---:|---:|---:|"]
    for left,right in zip(STAGES,STAGES[1:]):
        cells=exp["adjacent_layerwise_CKA"][left+"__to__"+right]
        dm=depth_diagnostics(grid(cells,"mean","whole_CKA"))
        dl=depth_diagnostics(grid(cells,"last","whole_CKA"))
        lines.append(f"| {left} → {right} | {fmt(dm['diag_mean'])} | "
                     f"{fmt(dl['diag_mean'])} | {fmt(dm['final'])} | "
                     f"{fmt(dl['final'])} | {len(dm['mutual'])}/5 |")
    lines += ["",
       "**Interpretation:** the mid2→RL transition retains high masked-mean",
       "overlap but much lower last-token correspondence, despite a large",
       "A/B response-format improvement. RL→mOPD has near-identical",
       "mean and last-token geometry with much smaller behavior changes.",
       "A blank/undefined embedding-last cell is not zero or a failed model.",
       "None of these observational correlations identifies causal training",
       "effects or rules out attention/rotary/prompt confounds.",
       "",
       "## Adjacent K2 layer overlap: complete matrices","",
       "Each matrix is **all 25** layer-pair CKA measurements.",
       "The best one-to-one alignment enumerates 5! possible assignments,",
       "and is a diagnostic of sampled representations, **not neuron homology**.",
       ""]
    for a,b in zip(STAGES,STAGES[1:]):
        edge=f"{a}__to__{b}";rows=exp["adjacent_layerwise_CKA"][edge]
        lines.extend((f"### {a} → {b}",""))
        for pooling in ("mean","last"):
            for label,metric in (("whole-statement","whole_CKA"),
                                  ("signed-role","signed_role_CKA")):
                m=grid(rows,pooling,metric)
                d=depth_diagnostics(m)
                row_best=", ".join(f"{K2[i]}→{K2[j] if j is not None else 'undefined'}" for i,j in enumerate(d["row_best"]))
                mutual=", ".join(f"{K2[i]}↔{K2[j]}" for i,j in d["mutual"])
                assignment=((", ".join(f"{K2[i]}→{K2[j] if j is not None else 'undefined'}" for i,j in
                              enumerate(d["optimal_assignment"]))) if d["optimal_assignment"] else "undefined")
                lines.extend((f"#### {pooling} pooled, {label}","",
                              markdown_grid(K2,K2,m),"",
                              f"Diagonal CKA mean **{fmt(d['diag_mean'])}**; final depth **{fmt(d['final'])}**.",
                              f"Each source layer's strongest target: {row_best}.",
                              f"Mutual nearest sampled layers: {mutual or 'none'}.",
                              f"Optimal 5×5 one-to-one assignment {assignment}; "
                              f"average **{fmt(d['optimal_mean'])}**; "
                              f"exact-depth row matches {d['diagonal_match_count']}/5.",""))
    lines += ["## K2↔Qwen cross-family matrices","",
       "Different backbone widths (K2 1536; Qwen 1024) and architectures;",
       "shared Gram geometry does not imply Qwen3.5 parent weights or teacher identity.",
       ""]
    for name in ("k2_pretrain_vs_qwen08","k2_release_vs_qwen08"):
        lines.extend((f"### {name}",""))
        for pooling in ("mean","last"):
            for label,metric in (("whole-statement","whole_cka"),
                                  ("signed-role","signed_role_cka")):
                m=grid(cross["cross_model_cka"][name]["comparisons"],
                       pooling,metric,K2,QWEN)
                lines.extend((f"#### {pooling}, {label}","",markdown_grid(K2,QWEN,m),""))
    lines += ["## Interpretation and limits","",
       "1. EXP-012/018 exact weight hashes, not CKA, establish pinned within-K2 checkpoint continuity.",
       "2. High cross-architecture CKA does not identify a training teacher or source weights.",
       "3. Multiple components changed between stages, including RoPE; these are observational comparisons.",
       "4. There are only 32 synthetic name-pair groups and eight dependent templates.",
       "5. EXP-018 proved mOPD and released main identical for all 255 BF16 tensors;",
       "   diagonal/optimal CKA 1.0 at matched depths is unsurprising.",
       "6. Independent human-adjudicated semantic gold, tokenizer/prompt controls,",
       "   actual teacher-trace custody and stage-wise causal ablations remain open.",
       "",
       "## Reproduce without regenerating models","",
       "```bash",
       "python3 research/k2_layer_overlap.py \\",
       "  --evidence-dir /home/scott/knowledge/50-Research/llm-research/latent-world-geometry-2026 \\",
       "  --out /home/scott/knowledge/50-Research/llm-research/latent-world-geometry-2026/EXPERIMENT-022-LAYER-OVERLAP.md",
       "```",
       "",
       "Fail closed if source digest/schema changes, a matrix has missing/duplicate cells,",
       "or the output already exists (unless explicitly `--replace`).",""]
    return "\n".join(lines)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evidence-dir",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--replace",action="store_true")
    a=p.parse_args()
    e=pinned(a.evidence_dir)
    doc=report(e)
    if a.out.exists() and not a.replace:raise SystemExit("STOP output already exists")
    if not a.out.parent.exists():raise SystemExit("STOP output directory missing")
    tmp=a.out.with_name(a.out.name+".tmp")
    if tmp.exists():raise SystemExit("STOP temporary path occupied")
    tmp.write_text(doc,encoding="utf8")
    tmp.replace(a.out)
    print("K2_LAYER_MATRIX_PASS",len(doc.splitlines()),"lines",
          hashlib.sha256(doc.encode()).hexdigest())

if __name__=="__main__":main()
