#!/usr/bin/env python3
"""Frozen prospective token-gap bounded EXP-054 decoy preflight.

No neural forwards or checkpoint loading, no model scores and no human semantic
foil claims. Original frozen sources; native tokenizer-only selection on a
fixed source-prefix word-boundary grid, never case filtering or substitution.
"""
import argparse
import hashlib
import json
import re
import statistics
from pathlib import Path

from exp053_agent_only import machine_receipt
from exp053_tokenizer_preflight import PRIVATE, PATHS, validate_custody
from exp053_agent_qwen_cpu import FROZEN, read_receipt, create_immutable, private_dir
from exp053_agent_k2_cpu import RECEIPTS
from exp054_prefix_length_control import prompt,options_for,digest

SCHEMA="exp054-token-gap-bounded-word-prefix-v1"
MIN_CHARS=256
CAP_CHARS=1600
MIN_RATIO_PER_MILLE=600
MAX_TOKEN_GAP=8
MODELS=("Qwen3.5-4B-Base","MiniCPM-V-4.7-1B","K2-Horizon-3.7B")
ARMS=("supported_prefix","decoy_prefix")


def hold(message):
    raise ValueError("HOLD EXP054 TOKEN-BALANCED: "+message)


def word_boundary_prefixes(text):
    if not isinstance(text,str):
        hold("invalid source input")
    return [m.end() for m in re.finditer(r'\S+\s*',text)
            if MIN_CHARS<=m.end()<=CAP_CHARS]


def prefix_token_grid(text,question,opts,tokenizers):
    output=[]
    for length in word_boundary_prefixes(text):
        s=prompt(question,opts,text[:length])
        counts=tuple(len(t.encode(s,add_special_tokens=True)) for t in tokenizers)
        if any(n<1 or n>4096 for n in counts):
            hold("token window violation")
        output.append((length,counts))
    if not output:
        hold("no source prefix candidates")
    return output


def choose(a,b):
    """Smallest max cross-tokenizer gap, then highest min source retention.

    No target labels, human answers, logits, generative model calls, or
    case-specific post-hoc thresholds influence this fixed ranking.
    """
    winner=None
    for len_a,tokens_a in a:
        for len_b,tokens_b in b:
            if min(len_a,len_b)*1000 < MIN_RATIO_PER_MILLE*max(len_a,len_b):
                continue
            gap=max(abs(x-y) for x,y in zip(tokens_a,tokens_b))
            if gap>MAX_TOKEN_GAP:
                continue
            key=(-gap,min(len_a,len_b),-abs(len_a-len_b),
                 max(len_a,len_b),-len_a,-len_b)
            if winner is None or key>winner[0]:
                winner=(key,len_a,len_b,tokens_a,tokens_b)
    if winner is None:
        hold("no compliant word-boundary token balance for frozen case")
    _,length_a,length_b,tokens_a,tokens_b=winner
    return length_a,length_b,tokens_a,tokens_b


def tokenizers_native():
    from transformers import AutoTokenizer
    checkpoint=json.loads(FROZEN.read_bytes())["models"]
    loaded=[]
    for name in MODELS:
        model_path=PATHS[name]
        if hashlib.sha256((model_path/"tokenizer.json").read_bytes()).hexdigest()!=checkpoint[name]["tokenizer_sha256"]:
            hold("tokenizer checkpoint identity mismatch")
        native=AutoTokenizer.from_pretrained(
            str(model_path),local_files_only=True,trust_remote_code=False)
        # Tokenizer-only use. Never call model.from_pretrained here.
        boundary={label:native.encode(" "+label,add_special_tokens=False)
                  for label in "ABCD"}
        if any(len(x)!=1 for x in boundary.values()) or len({x[0] for x in boundary.values()})!=4:
            hold("four answer labels require unique one-token continuation")
        loaded.append(native)
    return loaded


def build(lookup,admitted,tokenizers):
    if len(lookup)!=64 or admitted.get("human_approval") is not False or set(lookup)!=set(admitted.get("accepted",{})):
        hold("private frozen source/answer admission mismatch")
    ids=sorted(lookup)
    result=[]
    for i,cid in enumerate(ids):
        current=lookup[cid]
        decoy_id=ids[(i+1)%64]
        other=lookup[decoy_id]
        opts=options_for(current,0)
        a=prefix_token_grid(current["article"],current["question"],opts,tokenizers)
        b=prefix_token_grid(other["article"],current["question"],opts,tokenizers)
        la,lb,ta,tb=choose(a,b)
        prefixes={"supported_prefix":current["article"][:la],
                  "decoy_prefix":other["article"][:lb]}
        if prefixes["supported_prefix"]==prefixes["decoy_prefix"]:
            hold("unsupported identical supported and decoy")
        controls=[]
        max_gap=0
        for rotation in (0,1):
            rotated=options_for(current,rotation)
            for arm in ARMS:
                s=prompt(current["question"],rotated,prefixes[arm])
                tokens=[]
                for native in tokenizers:
                    ids_current=native.encode(s,add_special_tokens=True)
                    if not 1<=len(ids_current)<=4096:
                        hold("prompt token window drift")
                    for letter in "ABCD":
                        lbl=native.encode(" "+letter,add_special_tokens=False)
                        if native.encode(s+" "+letter,add_special_tokens=True)!=ids_current+lbl:
                            hold("one-token answer boundary drift")
                    tokens.append(len(ids_current))
                controls.append({"rotation":rotation,"arm":arm,
                                 "prompt_sha256":digest(s),
                                 "passage_sha256":digest(prefixes[arm]),
                                 "passage_codepoints":len(prefixes[arm]),
                                 "input_tokens":dict(zip(MODELS,tokens))})
            left=controls[-2]["input_tokens"]
            right=controls[-1]["input_tokens"]
            diff=max(abs(left[model]-right[model]) for model in MODELS)
            max_gap=max(max_gap,diff)
            if diff>MAX_TOKEN_GAP:
                hold("alternate rotation breaks tokenizer gap boundary")
        if len(controls)!=4 or len({x["prompt_sha256"] for x in controls})!=4:
            hold("bad token-balanced 4-prompt grid")
        result.append({"cohort_index":i,"case_id":cid,"decoy_case_id":decoy_id,
                       "supported_prefix_codepoints":la,"decoy_prefix_codepoints":lb,
                       "maximum_native_token_gap":max_gap,"conditions":controls})
    return result


def root_manifest(rows,packet_sha,code_sha):
    if len(rows)!=64:
        hold("missing or replaced frozen cases")
    return {"schema":SCHEMA,"human_adjudicated":False,
            "new_model_forward_passes":0,
            "source_packet_sha256":packet_sha,
            "case_count":64,"source_arms_per_rotation":2,"rotations":2,
            "prefix_cap_codepoints":CAP_CHARS,
            "minimum_prefix_codepoints":MIN_CHARS,
            "minimum_character_ratio_per_mille":MIN_RATIO_PER_MILLE,
            "maximum_native_token_count_gap":MAX_TOKEN_GAP,
            "native_models":list(MODELS),
            "algorithm_source_sha256":code_sha,
            "prompt_builder":"frozen exp053_tokenizer_preflight.prompt",
            "human_evidence_retained_verified":False,
            "semantic_grounding_proven":False,
            "model_training_overlap_excluded":False,
            "cases":rows,
            "limitations":"Tokenizer-only, approximate <=8 native tokens across all models and two rotations, not exact parity. Prefix may omit answer evidence, no human foil adjudication, no causal inference."}


def canonical_bytes(value):
    # Match the immutable private writer's actual newline-terminated JSON bytes.
    return (json.dumps(value,sort_keys=True,separators=(",",":"))+"\n").encode()


def publish_report(report):
    private_dir(RECEIPTS)
    chunks=[]
    for i in range(0,64,8):
        filename=f"exp054-wordprefix-balanced-part{i//8:02}-v1.json"
        payload={
            "schema":"exp054-wordprefix-balanced-part-v1",
            "source_packet_sha256":report["source_packet_sha256"],
            "algorithm_source_sha256":report["algorithm_source_sha256"],
            "starting_index":i,
            "case_count":8,
            "new_model_forward_passes":0,
            "cases":report["cases"][i:i+8]}
        data=canonical_bytes(payload)
        if len(data)>32768:
            hold("partition exceeds preexisting private writer limit")
        path=RECEIPTS/filename
        if path.exists() or path.is_symlink():
            if read_receipt(path)!=payload or path.read_bytes()!=data:
                hold("existing partition differs")
            checksum=hashlib.sha256(data).hexdigest()
        else:
            checksum=create_immutable(RECEIPTS,filename,payload)
        chunks.append({"filename":filename,"sha256":checksum,"starting_index":i,"case_count":8})
    index={k:v for k,v in report.items() if k!="cases"}
    index["schema"]="exp054-wordprefix-balanced-index-v1"
    index["parts"]=chunks
    index["complete_manifest_sha256"]=hashlib.sha256(canonical_bytes(report)).hexdigest()
    name="exp054-wordprefix-balanced-index-v1.json"
    path=RECEIPTS/name
    data=canonical_bytes(index)
    if path.exists() or path.is_symlink():
        if read_receipt(path)!=index or path.read_bytes()!=data:
            hold("existing token-balance index differs")
        checksum=hashlib.sha256(data).hexdigest()
    else:
        checksum=create_immutable(RECEIPTS,name,index)
    # Re-verify the immutable commit tree before success.
    recovered=[]
    for part in chunks:
        p=RECEIPTS/part["filename"]
        if hashlib.sha256(p.read_bytes()).hexdigest()!=part["sha256"]:
            hold("part checksum changed after publish")
        r=read_receipt(p)
        if r.get("source_packet_sha256")!=report["source_packet_sha256"]:
            hold("partition source packet mismatch")
        recovered.extend(r["cases"])
    copy=dict(report)
    copy["cases"]=recovered
    if len(recovered)!=64 or hashlib.sha256(canonical_bytes(copy)).hexdigest()!=index["complete_manifest_sha256"]:
        hold("private balanced manifest reconstruction mismatch")
    return checksum


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write-private",action="store_true")
    args=ap.parse_args()
    lookup,manifest=validate_custody()
    admitted=machine_receipt(PRIVATE.parent)
    natives=tokenizers_native()
    cases=build(lookup,admitted,natives)
    report=root_manifest(cases,manifest["packet_sha256"],hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    foil_path=RECEIPTS/"exp054-all-three-machine-foils-v1.json"
    frozen_foils=read_receipt(foil_path)
    if (frozen_foils.get("human_adjudicated") is not False or
        frozen_foils.get("based_on_target_model_scores") is not False or
        frozen_foils.get("exact_case_count")!=64 or
        frozen_foils.get("source_packet_sha256")!=manifest["packet_sha256"] or
        [x.get("case_id") for x in frozen_foils.get("cases",[])] != [x["case_id"] for x in cases]):
        hold("independent private three-foil precommit does not match all source cases")
    report["predeclared_machine_foils_sha256"]=hashlib.sha256(foil_path.read_bytes()).hexdigest()
    gaps=[x["maximum_native_token_gap"] for x in cases]
    lengths=[min(x["supported_prefix_codepoints"],x["decoy_prefix_codepoints"]) for x in cases]
    print("EXP054_WORDPREFIX_TOKEN_BALANCE_FEASIBILITY",{
        "source_cases":len(cases),"native_tokenizers":len(natives),
        "max_gap_across_rotations":max(gaps),
        "exact_native_all_model_case_matches":sum(x==0 for x in gaps),
        "within_2_native_tokens":sum(x<=2 for x in gaps),
        "within_4_native_tokens":sum(x<=4 for x in gaps),
        "prefix_min_codepoints":min(lengths),
        "prefix_mean_codepoints":round(statistics.mean(lengths),2),
        "new_model_forwards":0},flush=True)
    if args.write_private:
        checksum=publish_report(report)
        print("EXP054_WORDPREFIX_IMMUTABLE_COMMIT sha256="+checksum+" parts=8",flush=True)


if __name__=="__main__":
    main()
