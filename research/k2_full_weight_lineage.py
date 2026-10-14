#!/usr/bin/env python3
"""EXP025: K2 full 255-tensor BF16 checkpoint drift (including head and embedding).

Measurements are written only as Markdown in the central knowledge repository.
No model inference, training, production writes, or source-weight changes.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import k2_weight_drift as drift

EXTRAS=("model.embed_tokens.weight","model.norm.weight","lm_head.weight")
EXPECTED_TOTAL=1078285824
EXPECTED_LAYERS=880889856
EXPECTED_COS={
 "pretrain600k → mid1_75k":(0.9144990405579245,0.4337336506806616),
 "mid1_75k → mid2_47684":(0.9880153355360565,0.154414786669778),
 "mid2_47684 → rl_merged":(0.9999816387238073,0.001415872373186162),
 "rl_merged → mopd_249":(0.9999999391603563,7.755934915813537e-05),
}

def check_host_ram(min_gib):
    lines=Path("/proc/meminfo").read_text().splitlines()
    free=next(int(line.split()[1]) for line in lines if line.startswith("MemAvailable:"))
    if free < min_gib*1024*1024:
        raise RuntimeError("EXP025 resource HOLD: memory below "+str(min_gib)+"GiB")

def process(reference,model_root):
    checkpoints={stage:drift.open_manifest(model_root,reference,stage)
                 for stage in drift.STAGES}
    expected=set(checkpoints[drift.STAGES[0]]["tensor_to_file"])
    if len(expected)!=255 or any(set(v["tensor_to_file"])!=expected
                                  for v in checkpoints.values()):
        raise ValueError("Checkpoint 255-tensor inventory mismatch")
    layer_names={drift.tname(i,k) for i in range(28) for k in drift.KINDS}
    if len(layer_names)!=252 or expected-layer_names!=set(EXTRAS):
        raise ValueError("3 omitted tensor families not exactly enumerated")
    result={}
    for left,right in zip(drift.STAGES,drift.STAGES[1:]):
        edge=f"{left} → {right}"
        rows={}
        for name in sorted(expected):
            check_host_ram(22)
            rows[name]=drift.tensor_pair(checkpoints[left],checkpoints[right],name)
        layers=drift.merge([rows[name] for name in sorted(layer_names)])
        full=drift.merge([rows[name] for name in sorted(expected)])
        extras={name:drift.merge([rows[name]]) for name in EXTRAS}
        if (layers["n"],full["n"])!=(EXPECTED_LAYERS,EXPECTED_TOTAL):
            raise RuntimeError("Parameter count mismatch "+edge)
        if sum(v["n"] for v in extras.values())!=EXPECTED_TOTAL-EXPECTED_LAYERS:
            raise RuntimeError("Omitted parameter inventory mismatch")
        e_cos,e_l2=EXPECTED_COS[edge]
        if abs(layers["cosine"]-e_cos)>2e-8 or abs(layers["delta_l2"]-e_l2)>2e-8:
            raise RuntimeError("EXP023 direct-layer drift parity failed "+edge)
        # Norm-squared delta sums decompose by tensor disjoint union.
        sq_total=sum(v["nd"] for v in rows.values())
        sq_extra=sum(rows[name]["nd"] for name in EXTRAS)
        fractions={name:rows[name]["nd"]/sq_total if sq_total else 0.0
                   for name in EXTRAS}
        result[edge]={"layers":layers,"full":full,"extras":extras,
                      "delta_square_extras_fraction":sq_extra/sq_total if sq_total else 0,
                      "delta_square_fractions":fractions}
        print("EXP025_STAGE",edge,"full_l2",round(full["delta_l2"],9),
              "layer_l2",round(layers["delta_l2"],9),
              "extra_delta_fraction",round(result[edge]["delta_square_extras_fraction"],6),
              flush=True)
    return checkpoints,result

def display(checkpoints,measure,source_sha):
    out=["# EXP-025 — Full 255-tensor K2 checkpoint ancestry and weight drift",
         "",
         "**2026-10-14 | Actual read-only Safetensors measurement, 5 pinned snapshots.**",
         "Human-readable observations in central knowledge. No model inference.",
         "",
         "## Evidence custody and accounting",
         "",
         f"Pinned EXP-018 authority SHA256: `{source_sha}`.",
         "All 255 tensor names match across the five official checkpoint indices;",
         "BF16 type, shape, finiteness and scalar accounting verified at every pair.",
         f"Whole model parameters: **{EXPECTED_TOTAL:,}** across 255 tensors.",
         f"Transformer layers: **{EXPECTED_LAYERS:,}** across 252 tensors.",
         f"Embedding table, final normalization, and LM head: **{EXPECTED_TOTAL-EXPECTED_LAYERS:,}** across three tensors.",
         "",
         "| Checkpoint | Pinned HF revision | Model index SHA256 |",
         "|---|---|---|"]
    for stage in drift.STAGES:
        item=checkpoints[stage]
        out.append(f"| {stage} | `{item['revision']}` | `{item['index_sha256']}` |")
    out +=["","## Global weight drift: full 255 versus transformer-only 252",
          "",
          "Cosine and relative L2 are computed from original weight coordinates,",
          "weighted by tensor norms; BF16-change fraction counts individual values.",
          "",
          "| Transition | Full-model cosine | Full relative L2 drift | All changed BF16 values | Transformer-only relative L2 | Fraction of squared delta from 3 omitted tensors |",
          "|---|---:|---:|---:|---:|---:|"]
    for name,item in measure.items():
        all_=item["full"];prior=item["layers"]
        out.append(f"| {name} | {all_['cosine']:.9f} | {all_['delta_l2']:.8f} | "
                   f"{all_['changed_fraction']*100:.2f}% | "
                   f"{prior['delta_l2']:.8f} | "
                   f"{item['delta_square_extras_fraction']*100:.2f}% |")
    out +=["","## Missing three-tensor-family decomposition",
          "",
          "Embedding and untied output head have enormous parameter counts.",
          "A high raw number of changed BF16 values need not mean a large norm shift.",
          "The last column below is a contribution to whole-model squared",
          "weight *difference*, not a fraction of downstream causal behavior.",
          "",
          "| Transition | Family | Tensor parameters | Relative L2 | Changed BF16 fraction | Delta-square share | Exact identity |",
          "|---|---|---:|---:|---:|---:|"]
    names={"model.embed_tokens.weight":"Input token embeddings",
           "model.norm.weight":"Final transformer normalization",
           "lm_head.weight":"Untied output projection (LM head)"}
    for edge,item in measure.items():
        for name in EXTRAS:
            row=item["extras"][name]
            out.append(f"| {edge} | {names[name]} | {row['n']:,} | "
                       f"{row['delta_l2']:.8f} | "
                       f"{row['changed_fraction']*100:.2f}% | "
                       f"{item['delta_square_fractions'][name]*100:.2f}% | "
                       f"{row['same_tensors']}/1 |")
    out +=["","## Relation to K2 behavior and layer CKA",
          "",
          "EXP-023 deliberately measured only 252 block tensors.",
          "This experiment closes the three omitted model tensors and",
          "independently reproduces the EXP-023 transformer-only measurements.",
          "Whole-model weight proximity is stronger provenance evidence",
          "than hidden-state CKA, but neither weight cosine nor CKA",
          "identifies a training teacher. Model configuration changed from",
          "default RoPE to YaRN at the mid2→RL transition; EXP-024",
          "separately isolated those fixed-weight forward effects.",
          "These five checkpoints share architecture and tensor naming.",
          "Do not infer Qwen3.5 parameter ancestry from geometry or tasks.",
          "",
          "## Reproducibility and safety",
          "",
          "Script: scottjoyner/lms research/k2_full_weight_lineage.py.",
          "Protocol: EXPERIMENT-025-PROTOCOL.md under this knowledge directory.",
          "Requires pinned official checkpoint files (read-only),",
          "Torch and Safetensors CPU, and EXP-018 source receipt.",
          "Source files, model weights, graph access, credentials and NAS",
          "were never modified. Script fails closed on count, shape,",
          "source hash, resource pressure or prior direct-measurement mismatch.",
          "",
          "```bash",
          "python research/k2_full_weight_lineage.py --evidence-dir /home/scott/knowledge/50-Research/llm-research/latent-world-geometry-2026 --out /home/scott/knowledge/50-Research/llm-research/latent-world-geometry-2026/EXPERIMENT-025-FULL-WEIGHT-DRIFT.md",
          "```",
          ""]
    return "\n".join(out)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    a=parser.parse_args()
    if a.out.exists():raise SystemExit("STOP: output exists")
    check_host_ram(38)
    source=a.evidence_dir/"results_exp018_intermediate_k2_20261014.json"
    receipt=drift.sha_bytes(source)
    if receipt!=drift.REFERENCE_SHA:raise RuntimeError("Pinned EXP018 SHA drift")
    prev=json.loads(source.read_text())
    checkpoints,measures=process(prev,drift.ROOT)
    result=display(checkpoints,measures,receipt)
    if not a.out.parent.is_dir():raise RuntimeError("Report parent missing")
    tmp=a.out.with_name(a.out.name+".tmp")
    if tmp.exists():raise RuntimeError("Report scratch collision")
    tmp.write_text(result,encoding="utf8")
    tmp.replace(a.out)
    print("EXP025_PASS",len(result.splitlines()),hashlib.sha256(result.encode()).hexdigest(),flush=True)

if __name__=="__main__":main()
