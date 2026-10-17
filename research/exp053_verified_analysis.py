#!/usr/bin/env python3
"""EXP053 verified model-free scoring replay; no model imports or inference."""
import argparse,hashlib,json
from pathlib import Path
from exp053_race_review import MANIFEST,MANIFEST_SHA,digest,filehash,verified_admission,deny
from exp053_tokenizer_preflight import PRIVATE,PATHS,ROOT,WITHHELD,options_for,prompt
from exp053_metrics import analyze

PREFLIGHT=ROOT/"EXP053-TOKENIZER-PREFLIGHT-2026-10-16.json"
PREFLIGHT_SHA="8e35a44e1ebb725c0b52469baa4ab5d41472e38fe55346352b96dd26697726be"
PROMPT_CODE=ROOT/"exp053_tokenizer_preflight.py"
PROMPT_CODE_SHA="cf138946680db1c1259edff4ac895c4492992b69edb275d3c3c91e7e39f44664"
FIELDS=("case_id","article_sha256","model","arm","rotation",
        "option_order_sha256","prompt_sha256","context_article_sha256")

def sha(s):return hashlib.sha256(s.encode("utf8")).hexdigest()
def require_sources():
    if not PREFLIGHT.is_file() or filehash(PREFLIGHT)!=PREFLIGHT_SHA:
        deny("tokenizer feasibility receipt modified")
    if not PROMPT_CODE.is_file() or filehash(PROMPT_CODE)!=PROMPT_CODE_SHA:
        deny("prompt builder source drift")
    if not MANIFEST.is_file() or filehash(MANIFEST)!=MANIFEST_SHA:
        deny("frozen public manifest changed")
    scan=json.loads(PREFLIGHT.read_text())
    if set(scan["models"])!=set(PATHS) or scan["model_forward_passes"]!=0:
        deny("tokenizer feasibility report invalid")
    packet=json.loads(PRIVATE.read_text())
    manifest=json.loads(MANIFEST.read_text())
    if digest(packet)!=manifest["packet_sha256"]:
        deny("private packet does not match the public hash")
    return packet,manifest,scan

def check_private_archive(rows,approved,packet,scan):
    cases={c["case_id"]:c for c in packet["cases"]}
    ids=sorted(approved)
    if len(ids)<40:deny("fewer than 40 signed approved cases")
    next_case={c:ids[(i+1)%len(ids)] for i,c in enumerate(ids)}
    for record in rows:
        if not isinstance(record,dict) or any(k not in record for k in FIELDS):
            deny("scorer lacks required source receipt fields")
        cid=record["case_id"];rot=record["rotation"];arm=record["arm"]
        if cid not in approved or type(rot)!=int or rot not in (0,1):
            deny("unapproved or invalid scored case")
        if arm not in ("supported","withheld","decoy"):deny("invalid source arm")
        opts=options_for(cases[cid],rot)
        if record["option_order_sha256"]!=digest(opts):
            deny("displayed options do not match the frozen packet and rotation")
        article=(cases[cid]["article"] if arm=="supported" else
                 WITHHELD if arm=="withheld" else cases[next_case[cid]]["article"])
        if record["prompt_sha256"]!=sha(prompt(cases[cid]["question"],opts,article)):
            deny("prompt hash does not match source, question, options and decoy")
        model=record["model"]
        if model not in scan["models"]:deny("unknown model")
        if record["tokenizer_sha256"]!=scan["models"][model]["tokenizer_json_sha256"]:
            deny("tokenizer does not match the frozen preflight checkpoint")
    return len(rows)

def replay(private_dir,trust_roots,score_archive,output,adjudicator=None):
    if not output.parent.is_dir() or output.parent.resolve()!=ROOT.resolve() or output.exists():
        deny("public aggregate output must be a new file inside research")
    if not score_archive.resolve().is_relative_to(private_dir.resolve()):
        deny("individual score archive must be private")
    admission=verified_admission(private_dir,trust_roots,adjudicator)
    packet,manifest,scan=require_sources()
    if admission["packet_sha256"]!=digest(packet):deny("admission packet mismatch")
    rows=json.loads(score_archive.read_text())
    if not isinstance(rows,list):deny("score archive must be a JSON list")
    check_private_archive(rows,admission["accepted"],packet,scan)
    summary=analyze(rows,admission["accepted"],manifest)
    summary["score_archive_sha256"]=filehash(score_archive)
    summary["signed_admission_sha256"]=digest(admission)
    summary["tokenizer_preflight_sha256"]=PREFLIGHT_SHA
    summary["privacy"]="aggregates only; individual scores and RACE text stay private"
    with output.open("x") as f:f.write(json.dumps(summary,sort_keys=True,indent=2)+"\n")
    return summary

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--private-dir",type=Path,required=True)
    p.add_argument("--trust-roots",type=Path,required=True)
    p.add_argument("--private-score-archive",type=Path,required=True)
    p.add_argument("--adjudicator-file",type=Path)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(replay(a.private_dir,a.trust_roots,a.private_score_archive,
                            a.out,a.adjudicator_file),sort_keys=True))
if __name__=="__main__":main()
