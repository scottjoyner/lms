"""EXP050: gold/foil candidate sensitivity to actual vs unrelated same-paper question."""
import hashlib,json
from k2_qasper_matched_foil_fixture import make as original_matched,source_digest as matched_receipt
from k2_qasper_decode_fixture import cohort,prompt,receipt
BASE_SHA="16cd90495c18a1cb27492aca0a2ffb8b2def35eb6c615e7c952ed230af149b4d"
ORIGINAL_SHA="0f9c4fad7ba905a3b1283911ea50b77f4470b710183509425e24bd14328f69c5"
ARMS=("original","withheld")
KINDS=("supported","human_unanswerable")
MODES=("default","yarn")
def sha(x):return hashlib.sha256(x.encode("utf8")).hexdigest()
def build():
    base=original_matched()
    if matched_receipt(base)!=BASE_SHA or len(base)!=16:raise RuntimeError("HOLD frozen matched candidate")
    source=cohort()
    if receipt(source)!=ORIGINAL_SHA:raise RuntimeError("HOLD QASPER original source")
    unans={r["topic"]:r for r in source if r["kind"]=="human_unanswerable"}
    if len(unans)!=8:raise RuntimeError("HOLD unanswerable complement")
    out=[]
    for old in base:
        for kind in KINDS:
            row=dict(old["row"])
            if kind=="human_unanswerable":
                alt=unans[old["topic"]]
                if alt["pid"]!=row["pid"] or alt["question"]==row["question"]:
                    raise RuntimeError("HOLD question-swap wrong source")
                row["question"]=alt["question"]
                row["qid"]=alt["qid"]
                row["gold"]=alt["gold"]
            out.append({"topic":old["topic"],"arm":old["arm"],"kind":kind,
               "row":row,"gold":old["candidate_gold"],"foil":old["candidate_foil"],
               "n":old["n"]})
    if len(out)!=32:raise RuntimeError("HOLD wrong study count")
    for i in range(0,32,2):
        a,b=out[i:i+2]
        if (a["kind"],b["kind"])!=KINDS or a["row"]["context"]!=b["row"]["context"]:
            raise RuntimeError("HOLD paired same-evidence question condition")
        if a["gold"]!=b["gold"] or a["foil"]!=b["foil"]:
            raise RuntimeError("HOLD question-swap changed candidates")
    return out
def digest(rows=None):
    rows=build() if rows is None else rows
    data=[{"topic":r["topic"],"arm":r["arm"],"kind":r["kind"],
      "prompt":prompt(r["row"]),"gold_candidate":r["gold"],
      "foil_candidate":r["foil"],"target_tokens":r["n"]} for r in rows]
    return sha(json.dumps(data,sort_keys=True,ensure_ascii=False,separators=(",",":")))
if __name__=="__main__":
    rows=build()
    print("EXP050_COEFFICIENT_COHORT",len(rows),"TOPICS",
      len({r["topic"] for r in rows}),"SHA256",digest(rows))
