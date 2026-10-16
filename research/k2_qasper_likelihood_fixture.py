"""EXP048: exact frozen QASPER 8-paper questions and two teacher-forced candidates.

Scoring candidates is NOT free-generation success; this is a within-source
diagnostic of response recognition and evidence sensitivity.
"""
import hashlib,json
from k2_qasper_decode_fixture import cohort,prompt,receipt

ORIGINAL_COHORT_SHA="0f9c4fad7ba905a3b1283911ea50b77f4470b710183509425e24bd14328f69c5"
MISSING="The research-paper excerpt was withheld."
CANDIDATE_UNKNOWN=" UNKNOWN"
CATEGORIES=("supported","human_unanswerable")
ARMS=("original","withheld")
MODES=("default","yarn")

def sha(t):return hashlib.sha256(t.encode("utf8")).hexdigest()

def inputs():
    source=cohort()
    if receipt(source)!=ORIGINAL_COHORT_SHA or len(source)!=16:
        raise RuntimeError("HOLD EXP047 source drift")
    supported={r["topic"]:r["gold"] for r in source if r["kind"]=="supported"}
    if len(supported)!=8 or any(not v or v=="UNKNOWN" for v in supported.values()):
        raise RuntimeError("HOLD invalid gold candidates")
    out=[]
    for row in source:
        foil=supported[row["topic"]]
        if row["kind"]=="supported" and row["gold"]!=foil:
            raise RuntimeError("HOLD wrong supported answer candidate")
        if row["kind"]=="human_unanswerable" and row["gold"]!="UNKNOWN":
            raise RuntimeError("HOLD human answerability label drift")
        for arm in ARMS:
            r=dict(row)
            r["context"]=row["context"] if arm=="original" else MISSING
            out.append({"row":r,"kind":row["kind"],"topic":row["topic"],
                        "arm":arm,"choice_answer":" "+foil,
                        "choice_unknown":CANDIDATE_UNKNOWN})
    if len(out)!=32 or len({x["topic"] for x in out})!=8:
        raise RuntimeError("HOLD 32 frozen contexts")
    return out

def digest(items=None):
    items=inputs() if items is None else items
    data=[{"topic":v["topic"],"kind":v["kind"],"arm":v["arm"],
           "prompt":prompt(v["row"]),"A":v["choice_answer"],
           "B":v["choice_unknown"],"human_gold":v["row"]["gold"]} for v in items]
    return sha(json.dumps({"exp047":ORIGINAL_COHORT_SHA,"rows":data},
                          ensure_ascii=False,sort_keys=True,separators=(",",":")))

def token_inventory(tok,items):
    tokens=[]
    for item in items:
        prefix=tok.encode(prompt(item["row"]),add_special_tokens=True)
        answer=tok.encode(item["choice_answer"],add_special_tokens=False)
        unknown=tok.encode(item["choice_unknown"],add_special_tokens=False)
        if not (20<len(prefix)<750 and 1<=len(answer)<=32 and 1<=len(unknown)<=12):
            raise RuntimeError("HOLD invalid full tokenized candidate")
        if answer==unknown or any(t<0 for t in prefix+answer+unknown):
            raise RuntimeError("HOLD degenerate candidate IDs")
        tokens.append((prefix,answer,unknown))
    return tokens

if __name__=="__main__":
    a=inputs()
    print("EXP048_CONTEXTS",len(a),"COHORT_SHA",digest(a),
          "HUMAN_LABELS",{k:sum(x["kind"]==k for x in a) for k in CATEGORIES})
