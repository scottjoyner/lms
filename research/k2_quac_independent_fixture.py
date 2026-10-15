#!/usr/bin/env python3
"""EXP043: deterministic QuAC v0.2 validation QA, article-disjoint from SQuAD/ledger.

QuAC is human-annotated public data, NOT model-training-unseen. Synthetic
missing/swapped-context controls are separately tagged and are NOT QuAC labels.
Research only; source file downloaded externally and SHA-verified offline.
"""
import hashlib,json,re
from pathlib import Path

SOURCE=Path(__file__).resolve().parent/"data/quac-val-v0.2.json"
SOURCE_SHA="09e622916280ba04c9352acb1bc5bbe80f11a2598f6f34e934c51d9e6570f378"
COHORT_SIZE=12
CONDITIONS=("answerable","unanswerable","withheld","swapped")
GOLD_UNKNOWN="UNKNOWN"

def digest(t):return hashlib.sha256(t.encode("utf8")).hexdigest()
def source():
    if not SOURCE.is_file() or hashlib.sha256(SOURCE.read_bytes()).hexdigest()!=SOURCE_SHA:
        raise RuntimeError("HOLD official QuAC validation checksum")
    data=json.loads(SOURCE.read_text(encoding="utf8"))
    if len(data.get("data",[]))!=1000:
        raise RuntimeError("HOLD QuAC v0.2 validation inventory")
    return data["data"]

def get_history(qas,j):
    lines=[]
    for q in qas[max(0,j-2):j]:
        a=q.get("orig_answer",{}).get("text","").strip()
        if not a:raise RuntimeError("HOLD broken QuAC history answer")
        lines.append(("Q: "+q["question"].strip()+"\nA: "+("UNKNOWN" if a=="CANNOTANSWER" else a)))
    return "\n".join(lines)

def groups():
    eligible=[]
    for a in source():
        paragraphs=a.get("paragraphs",[])
        if len(paragraphs)!=1:continue
        p=paragraphs[0]
        ctx=p["context"].strip()
        if not 300<len(ctx)<1800:continue
        qas=p["qas"]
        ans=[];unknown=[]
        for j,q in enumerate(qas):
            txt=q["question"].strip()
            if not (12<len(txt)<145 and txt.lower().startswith(("who ","what ","where ","when ","which ","how "))):
                continue
            label=q.get("orig_answer",{}).get("text","").strip()
            if label=="CANNOTANSWER":
                unknown.append((j,q))
            elif 2<=len(label)<=40 and len(label.split())<=7 and label in ctx:
                if label.casefold() not in get_history(qas,j).casefold():
                    ans.append((j,q))
        if not ans or not unknown:continue
        # One paragraph/dialog per article title, original human labels only.
        k=digest(a["title"]+"|"+p["id"])
        eligible.append((k,a,p,ans,unknown))
    selected=[];titles=set()
    for _,a,p,ans,unknown in sorted(eligible,key=lambda t:t[0]):
        if a["title"] in titles:continue
        titles.add(a["title"])
        # Selection is deterministic and independent of model performance.
        ja,qa=min(ans,key=lambda t:digest(t[1]["id"]))
        ju,qu=min(unknown,key=lambda t:digest(t[1]["id"]))
        selected.append((a,p,ja,qa,ju,qu))
        if len(selected)==COHORT_SIZE:break
    if len(selected)!=COHORT_SIZE:raise RuntimeError("HOLD insufficient articles")
    return selected

def prompt(row):
    # Fixed prompt: actual QuAC dialog history shown to disambiguate questions.
    h=row["history"]
    return ("Read ONLY the provided Wikipedia passage and dialog history. "
       "Answer the current question with the shortest supported quotation from "
       "the passage, or write exactly UNKNOWN if it is not answerable from the "
       "provided evidence. Write one answer line and stop.\n"
       "Article: "+row["title"]+"\nSection: "+row["section"]+"\n"
       "Passage:\n"+row["context"]+"\n"+
       ("Prior dialog:\n"+h+"\n" if h else "")+
       "Current question: "+row["question"]+"\nAnswer:")

def build():
    selected=groups()
    rows=[]
    for topic,(a,p,ja,qa,ju,qu) in enumerate(selected):
        for kind,j,q in (("answerable",ja,qa),("unanswerable",ju,qu)):
            ref=q["orig_answer"]["text"].strip()
            rows.append({"topic":topic,"kind":kind,
               "title":a["title"],"section":a["section_title"],
               "context":p["context"].strip(),
               "history":get_history(p["qas"],j),
               "question":q["question"].strip(),
               "gold":ref if kind=="answerable" else GOLD_UNKNOWN,
               "qid":q["id"]})
        anchor=next(x for x in rows if x["topic"]==topic and x["kind"]=="answerable")
        # Six source-withheld and six cross-topic context swaps, from answerables;
        # operational abstention controls, not QuAC human no-answer annotations.
        if topic<6:
            rows.append({**anchor,"kind":"withheld",
                "context":"The source passage is withheld.",
                "history":"","gold":GOLD_UNKNOWN})
        else:
            dst=selected[(topic+1)%COHORT_SIZE][1]["context"].strip()
            if anchor["gold"].casefold() in dst.casefold():
                raise RuntimeError("HOLD source-swap gold leakage")
            rows.append({**anchor,"kind":"swapped",
                "context":dst,"history":"","gold":GOLD_UNKNOWN})
    shape={k:sum(x["kind"]==k for x in rows) for k in CONDITIONS}
    if shape!={"answerable":12,"unanswerable":12,"withheld":6,"swapped":6}:
        raise RuntimeError("HOLD stratification")
    if len({x["title"] for x in rows})!=COHORT_SIZE:
        raise RuntimeError("HOLD duplicate article title")
    return rows

def receipt(rows=None):
    if rows is None:rows=build()
    payload=[{"kind":r["kind"],"topic":r["topic"],"qid":r["qid"],
       "gold":r["gold"],"prompt":prompt(r)} for r in rows]
    return digest(json.dumps(payload,sort_keys=True,ensure_ascii=False,separators=(",",":")))

def normalize(t):
    # QuAC span exactness and grounded abstention with fixed punctuation/case.
    return re.sub(r"^[\W_]+|[\W_]+$","", " ".join(t.casefold().split()))

def score(row,completion,hit_cap=False,hit_newline=False):
    full=normalize(completion)
    first=normalize(completion.split("\n",1)[0])
    gold=normalize(row["gold"])
    exact=int(full==gold)
    first_exact=int(first==gold)
    # Token F1 (descriptive for human extractive answers, not official QuAC).
    a=full.split();b=gold.split()
    from collections import Counter
    overlap=sum((Counter(a)&Counter(b)).values())
    f1=2*overlap/(len(a)+len(b)) if (a and b) else 0.
    is_unknown=int(full=="unknown")
    contains=int(bool(re.search(r"(?<!\w)"+re.escape(gold)+r"(?!\w)",full)))
    return {"kind":row["kind"],"strict":exact,"first_line":first_exact,
      "f1":f1,"contains":contains,"unknown":is_unknown,
      "cap":int(hit_cap),"newline":int(hit_newline),"empty":int(not full)}

def aggregate(rows,outs):
    if len(rows)!=len(outs):raise RuntimeError("HOLD missing generations")
    totals={}
    for kind in CONDITIONS:
        chosen=[out for inp,out in zip(rows,outs) if inp["kind"]==kind]
        wanted={"answerable":12,"unanswerable":12,"withheld":6,"swapped":6}[kind]
        if len(chosen)!=wanted or any(x["kind"]!=kind for x in chosen):
            raise RuntimeError("HOLD incorrect output stratum")
        totals[kind]={"n":wanted}
        for field in ("strict","first_line","contains","unknown","cap","newline","empty"):
            totals[kind][field]=sum(int(x[field]) for x in chosen)
        totals[kind]["mean_f1"]=round(sum(float(x["f1"]) for x in chosen)/wanted,6)
    return totals

if __name__=="__main__":
    cases=build()
    print("EXP043_QUAC_SOURCE_SHA",SOURCE_SHA)
    print("EXP043_CASES",len(cases),"articles",len(set(r["title"] for r in cases)),
          "case-counts",{k:sum(r["kind"]==k for r in cases) for k in CONDITIONS})
    print("EXP043_COHORT_DIGEST",receipt(cases))
    print("Only metadata/digests emitted. No human-annotated passages printed.")
