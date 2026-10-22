#!/usr/bin/env python3
"""EXP044: remove QuAC's terminal CANNOTANSWER sentinel, preserve every QA label."""
import hashlib,json
import k2_quac_independent_fixture as q

ORIGINAL_SHA="28a344fe977ff0e3b10a4cd8707e2f00e9abf3f8bb875a6c9f65c4ccbce98fc1"
SENTINEL="CANNOTANSWER"
AFFECTED=30

def pairs():
    original=q.build()
    if q.receipt(original)!=ORIGINAL_SHA:
        raise RuntimeError("HOLD original QuAC selection changed")
    updated=[]
    changed=0
    for old in original:
        x=dict(old)
        source=x["context"]
        has=SENTINEL in source
        if x["kind"]=="withheld":
            if has:raise RuntimeError("HOLD withheld source has QuAC sentinel")
        else:
            if source.count(SENTINEL)!=1 or not source.endswith(SENTINEL):
                raise RuntimeError("HOLD QuAC sentinel not uniquely terminal")
            x["context"]=source[:-len(SENTINEL)].rstrip()
            if SENTINEL in x["context"]:
                raise RuntimeError("HOLD leftover source sentinel")
            changed+=1
        for key in ("topic","kind","title","section","history","question","gold","qid"):
            if x[key]!=old[key]:raise RuntimeError("HOLD modified non-context field")
        if x["kind"]=="answerable" and x["gold"] not in x["context"]:
            raise RuntimeError("HOLD stripped original ground-truth extract")
        if x["kind"]=="swapped" and x["gold"].casefold() in x["context"].casefold():
            raise RuntimeError("HOLD swapped gold contamination")
        updated.append(x)
    if len(original)!=36 or len(updated)!=36 or changed!=AFFECTED:
        raise RuntimeError("HOLD original QuAC cohort or sentinel count")
    if [x["context"] for x in original]==[x["context"] for x in updated]:
        raise RuntimeError("HOLD no actual source intervention")
    return original,updated

def pair_digest():
    original,cleaned=pairs()
    info=[
      {"topic":a["topic"],"qid":a["qid"],"kind":a["kind"],"gold":a["gold"],
       "original_prompt":q.prompt(a),"cleaned_prompt":q.prompt(b)}
      for a,b in zip(original,cleaned)]
    encoded=json.dumps(info,ensure_ascii=False,sort_keys=True,separators=(",",":"))
    return hashlib.sha256(encoded.encode()).hexdigest()

if __name__=="__main__":
    original,cleaned=pairs()
    print("EXP044_SENTINEL_COUNT",AFFECTED,"of",len(cleaned),"cohort")
    print("EXP044_ORIGINAL_PROMPT_SHA",q.receipt(original))
    print("EXP044_CLEAN_PROMPT_SHA",q.receipt(cleaned))
    print("EXP044_PAIRED_SHA",pair_digest())
    print("Only aggregate hashes; no QuAC source paragraphs or text samples emitted.")
