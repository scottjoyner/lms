#!/usr/bin/env python3
"""EXP054 precommit of equal-character-budget paired article-prefix controls.

No neural forwards, target logits, generative calls, or semantic label claims.
Both supported and unrelated decoy use the identical deterministic first-N
Unicode codepoints, where N=min(full article codepoint lengths, cap).
The resulting text is NOT guaranteed to retain the necessary answer evidence.
"""
import argparse
import hashlib
import json
from pathlib import Path

from exp053_agent_only import machine_receipt
from exp053_tokenizer_preflight import PRIVATE, validate_custody, prompt, options_for
from exp053_agent_k2_cpu import RECEIPTS, private_dir, create_immutable, read_receipt

SCHEMA="exp054-prefix-codepoint-paired-control-v1"
CODEPOINT_CAP=1600
MIN_CODEPOINTS=256
ARMS=("supported_prefix","decoy_prefix")
ROTATIONS=(0,1)


def hold(message):
    raise ValueError("HOLD EXP054 SOURCE-LENGTH: "+message)


def digest(data):
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def make_pair(case,decoy,limit=CODEPOINT_CAP):
    if type(limit) is not int or not MIN_CODEPOINTS<=limit<=CODEPOINT_CAP:
        hold("invalid frozen passage codepoint budget")
    articles=[case.get("article"),decoy.get("article")]
    if not all(isinstance(x,str) for x in articles):
        hold("invalid source text")
    n=min(limit,*(len(text) for text in articles))
    if n<MIN_CODEPOINTS:
        hold("short source articles cannot be silently replaced")
    a,b=(text[:n] for text in articles)
    if len(a)!=len(b) or len(a)!=n or a==b:
        hold("missing or identical decoy source prefix")
    return {"supported_prefix":a,"decoy_prefix":b,"codepoints":n}


def plan(lookup,admission):
    if not isinstance(lookup,dict) or len(lookup)!=64:
        hold("frozen source population incomplete")
    accepted=admission.get("accepted",{}) if isinstance(admission,dict) else {}
    if set(lookup)!=set(accepted) or admission.get("human_approval") is not False:
        hold("source/key admission mismatch")
    ids=sorted(lookup)
    results=[]
    for i,cid in enumerate(ids):
        this=lookup[cid]
        if (not isinstance(this,dict) or
            not isinstance(this.get("question"),str) or
            not this["question"].strip() or
            not isinstance(this.get("options"),dict) or
            set(this["options"]) != set("ABCD") or
            any(not isinstance(this["options"][k],str) or
                not this["options"][k].strip() for k in "ABCD") or
            len({this["options"][k].strip().casefold() for k in "ABCD"}) != 4):
            hold("malformed frozen question and answer choices")
        next_cid=ids[(i+1)%len(ids)]
        control=make_pair(this,lookup[next_cid])
        if len(control["supported_prefix"])!=len(control["decoy_prefix"]):
            hold("paired character budgets differ")
        if control["codepoints"]>CODEPOINT_CAP:
            hold("cap overflow")
        rows=[]
        for r in ROTATIONS:
            opts=options_for(this,r)
            for arm in ARMS:
                text=prompt(this["question"],opts,control[arm])
                if "Answer:" not in text or len(text)<=control["codepoints"]:
                    hold("unexpected prompt construction")
                rows.append({"rotation":r,"arm":arm,"prompt_sha256":digest(text),
                             "passage_sha256":digest(control[arm]),
                             "passage_codepoints":control["codepoints"],
                             "prompt_utf8_bytes":len(text.encode("utf-8"))})
        if len(rows)!=4 or len({r["prompt_sha256"] for r in rows})!=4:
            hold("prompt collision across source/rotation")
        results.append({"case_id":cid,
                        "decoy_case_id":next_cid,
                        "source_passage_sha256":digest(this["article"]),
                        "decoy_passage_sha256":digest(lookup[next_cid]["article"]),
                        "codepoint_budget":control["codepoints"],
                        "controls":rows})
    if len(results)!=64:
        hold("missing frozen source pair")
    return results


def build_manifest(lookup,admission,manifest_digest):
    rows=plan(lookup,admission)
    return {
      "schema":SCHEMA,
      "cohort_source_packet_sha256":manifest_digest,
      "source_case_count":64,
      "source_arms_per_rotation":2,
      "rotations":2,
      "prompts_per_model":256,
      "length_control_kind":"equal Unicode codepoint budget across supported and decoy within each article",
      "max_article_prefix_codepoints":CODEPOINT_CAP,
      "minimum_article_prefix_codepoints":MIN_CODEPOINTS,
      "token_length_parity_verified":False,
      "semantic_answer_evidence_retained_verified":False,
      "new_model_forward_passes":0,
      "human_adjudicated":False,
      "generation_or_abstention_proven":False,
      "possible_training_contamination_excluded":False,
      "foil_selection":"all three alternatives independently frozen in exp054-all-three-machine-foils-v1",
      "cases":rows,
      "limitations":"Equal codepoint budgets are not token-level parity; prefix truncation may remove answer evidence and cannot establish causal grounding."
    }


def canonical_bytes(value):
    return (json.dumps(value,sort_keys=True,separators=(",",":"))+"\\n").encode()


def verify_or_publish(filename,payload):
    path=RECEIPTS / filename
    if path.is_file() or path.is_symlink():
        if read_receipt(path)!=payload or path.read_bytes()!=canonical_bytes(payload):
            hold("existing immutable manifest shard differs")
        return hashlib.sha256(path.read_bytes()).hexdigest()
    return create_immutable(RECEIPTS,filename,payload)


def publish_partitioned_manifest(result):
    # Writer's existing maximum is 32768 bytes, deliberately not increased.
    if len(result["cases"])!=64:
        hold("source cohort missing")
    private_dir(RECEIPTS)
    preflight_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    parts=[]
    for start in range(0,64,16):
        payload={
            "schema":"exp054-prefix-codepoint-paired-control-part-v1",
            "source_schema":SCHEMA,
            "source_packet_sha256":result["cohort_source_packet_sha256"],
            "preflight_code_sha256":preflight_hash,
            "source_index_start":start,
            "source_case_count":16,
            "new_model_forward_passes":0,
            "cases":result["cases"][start:start+16],
        }
        name=f"exp054-prefix-length-control-part{start//16:02}-v1.json"
        if len(canonical_bytes(payload))>32768:
            hold("precommitted manifest part would exceed private size bound")
        parts.append({"filename":name,"sha256":verify_or_publish(name,payload),
                      "start":start,"cases":16})
    index={k:v for k,v in result.items() if k!="cases"}
    index.update({"schema":"exp054-prefix-codepoint-paired-control-index-v1",
                  "source_schema":SCHEMA,
                  "parts":parts,
                  "manifest_payload_sha256":hashlib.sha256(canonical_bytes(result)).hexdigest(),
                  "preflight_code_sha256":preflight_hash,
                  "new_model_forward_passes":0})
    digest=verify_or_publish("exp054-prefix-length-control-index-v1.json",index)
    # Verify complete topology after writing before reporting success.
    verify_parts(index,result)
    return digest


def verify_parts(index,expected):
    if index.get("source_case_count")!=64 or index.get("new_model_forward_passes")!=0:
        hold("invalid length-control index")
    parts=index.get("parts")
    if not isinstance(parts,list) or len(parts)!=4:
        hold("incomplete manifest shards")
    items=[]
    for i,part in enumerate(parts):
        name=f"exp054-prefix-length-control-part{i:02}-v1.json"
        if part.get("filename")!=name or part.get("start")!=16*i or part.get("cases")!=16:
            hold("invalid manifest partition")
        path=RECEIPTS/name
        blob=path.read_bytes() if path.is_file() else b""
        if hashlib.sha256(blob).hexdigest()!=part.get("sha256"):
            hold("manifest shard SHA mismatch")
        content=read_receipt(path)
        if content.get("source_index_start")!=i*16 or content.get("source_case_count")!=16:
            hold("manifest shard claims differ")
        if content.get("source_packet_sha256")!=index.get("cohort_source_packet_sha256"):
            hold("manifest source digest mismatch")
        items.extend(content["cases"])
    reconstructed=dict(expected)
    reconstructed["cases"]=items
    if len(items)!=64 or hashlib.sha256(canonical_bytes(reconstructed)).hexdigest()!=index.get("manifest_payload_sha256"):
        hold("complete manifest reconstruction differs")
    return True


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-private",action="store_true")
    args=parser.parse_args()
    lookup,manifest=validate_custody()
    admitted=machine_receipt(PRIVATE.parent)
    result=build_manifest(lookup,admitted,manifest["packet_sha256"])
    result["preflight_code_sha256"]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if args.write_private:
        out=publish_partitioned_manifest(result)
        print("EXP054_PRIVATE_LENGTH_CONTROL_FROZEN_INDEX sha256="+out+" shards=4",flush=True)
    codepoint_budgets=[row["codepoint_budget"] for row in result["cases"]]
    print("EXP054_CODEPOINT_PAIR_PREFLIGHT_OK",{
        "articles":len(codepoint_budgets),
        "prompts_per_model":256,
        "min_common_prefix_codepoints":min(codepoint_budgets),
        "max_common_prefix_codepoints":max(codepoint_budgets),
        "new_model_forwards":0,
        "token_length_parity_verified":False,
    },flush=True)


if __name__=="__main__":
    main()
