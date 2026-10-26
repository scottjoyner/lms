#!/usr/bin/env python3
"""EXP045: public human QASPER valid, new domain, evidence/decoy controls.

Uses one independently authored human answerable and one human-unanswerable
question per paper. Decoy/withheld context conditions are NOT human labels.
"""
import hashlib,json,re
from pathlib import Path
SOURCE=Path(__file__).resolve().parent/"data/qasper-dev-v0.3.json"
SOURCE_SHA="2ae7ee62a65b1c4225791c70de80c2aad4e8998cf1fd4f09a53103db4f21af93"
ARCHIVE_SHA="a28fdf966db827bcee3d873107d6b6669864fb7ca8fbf73a192f5e39191bdb5a"
KINDS=("supported","human_unanswerable","answerable_decoy","withheld")
N_PAPERS=12
SEED="EXP045-QASPER-deterministic-paper-sha-selection-20261015"
STOP={"the","an","of","is","in","a","what","which","are","does","do","did","for",
      "and","was","were","this","that","paper","method","how","with","they",
      "to","on","from","by","there","their","using","used","we","have","has"}

def sha(t):return hashlib.sha256(t.encode("utf8")).hexdigest()
def dataset():
    if not SOURCE.is_file() or hashlib.sha256(SOURCE.read_bytes()).hexdigest()!=SOURCE_SHA:
        raise RuntimeError("HOLD QASPER official v0.3 dev SHA")
    d=json.loads(SOURCE.read_text())
    if len(d)!=281:raise RuntimeError("HOLD QASPER paper inventory")
    return d

def lexical(t):
    return set(re.findall(r"[a-z0-9]{3,}",t.casefold()))-STOP

def candidates():
    d=dataset();eligible=[]
    for pid,p in d.items():
        paragraphs=[txt.strip() for sec in p["full_text"] for txt in sec["paragraphs"]
                  if isinstance(txt,str) and 100<len(txt.strip())<1050]
        if not paragraphs:continue
        supported=[];absent=[]
        for qa in p["qas"]:
            q=qa["question"].strip()
            if not (12<len(q)<160):continue
            anns=[a["answer"] for a in qa["answers"]]
            if not anns:continue
            if all(a.get("unanswerable") is True for a in anns):
                absent.append(qa)
            if not all(a.get("unanswerable") is False for a in anns):
                continue
            spans=[sp.strip() for an in anns for sp in an.get("extractive_spans",[])
                   if isinstance(sp,str) and 3<=len(sp.strip())<=65 and len(sp.split())<=9]
            if not spans:continue
            for gold in sorted(set(spans),key=lambda x:(sha(q+x),x)):
                evidence=[ev.strip() for an in anns for ev in an.get("evidence",[])
                      if isinstance(ev,str) and 100<len(ev.strip())<1050 and gold in ev]
                if evidence:
                    supported.append((qa,gold,sorted(set(evidence),key=sha)[0]))
                    break
        if not supported or not absent:continue
        ans=sorted(supported,key=lambda x:sha(pid+SEED+x[0]["question_id"]))[0]
        unknown=sorted(absent,key=lambda x:sha(pid+SEED+x["question_id"]))[0]
        # Same-paper, lexically related but not answer-containing paragraph.
        tokens=lexical(ans[0]["question"])
        scored=[]
        for txt in paragraphs:
            if ans[1].casefold() in txt.casefold() or txt==ans[2]:
                continue
            overlap=len(tokens&lexical(txt))
            if overlap>=1:scored.append((-overlap,sha(pid+SEED+txt),txt))
        if not scored:continue
        decoy=sorted(scored)[0][2]
        unknown_tokens=lexical(unknown["question"])
        ucandidates=[]
        for txt in paragraphs:
            overlap=len(unknown_tokens&lexical(txt))
            if overlap>=1:
                ucandidates.append((-overlap,sha(pid+SEED+txt),txt))
        if not ucandidates:continue
        unknown_context=sorted(ucandidates)[0][2]
        eligible.append((sha(SEED+pid),pid,p,ans,unknown,decoy,unknown_context))
    return sorted(eligible,key=lambda v:v[0])

def build():
    chosen=candidates()[:N_PAPERS]
    if len(chosen)!=N_PAPERS:raise RuntimeError("HOLD <12 eligible scholarly papers")
    rows=[]
    for topic,(_,pid,p,answerable,unk,decoy,unknown_context) in enumerate(chosen):
        qa,gold,evidence=answerable
        title=p["title"].strip()
        original={"topic":topic,"pid":pid,"title":title,"qid":qa["question_id"],
           "question":qa["question"].strip(),"gold":gold,"context":evidence,"kind":"supported"}
        rows.append(original)
        rows.append({"topic":topic,"pid":pid,"title":title,
            "qid":unk["question_id"],"question":unk["question"].strip(),
            "gold":"UNKNOWN","context":unknown_context,"kind":"human_unanswerable"})
        rows.append({**original,"kind":"answerable_decoy","context":decoy,"gold":"UNKNOWN"})
        rows.append({**original,"kind":"withheld","context":"The passage was withheld.","gold":"UNKNOWN"})
    if len(rows)!=48 or len(set(x["pid"] for x in rows))!=12:
        raise RuntimeError("HOLD QASPER source cohort")
    for row in rows:
        if row["kind"]=="supported" and row["gold"] not in row["context"]:
            raise RuntimeError("HOLD gold not in human evidence")
        if row["kind"]=="answerable_decoy":
            gold=next(y["gold"] for y in rows if y["topic"]==row["topic"] and y["kind"]=="supported")
            if gold.casefold() in row["context"].casefold():
                raise RuntimeError("HOLD gold leaks into decoy")
        if row["kind"]=="withheld" and row["context"]!="The passage was withheld.":
            raise RuntimeError("HOLD incorrect zero-source control")
    return rows

def prompt(r):
    return ("Answer using ONLY the provided research-paper excerpt. "
            "Use the shortest exact answer phrase supported by the excerpt, "
            "or exactly UNKNOWN if the excerpt does not support an answer. "
            "Write one line and stop.\n"
            "Paper: "+r["title"]+"\nExcerpt:\n"+r["context"]+
            "\nQuestion: "+r["question"]+"\nAnswer:")

def receipt(rows=None):
    if rows is None:rows=build()
    data=[{"topic":r["topic"],"pid":r["pid"],"qid":r["qid"],"kind":r["kind"],
       "gold":r["gold"],"prompt":prompt(r)} for r in rows]
    return sha(json.dumps(data,sort_keys=True,ensure_ascii=False,separators=(",",":")))

def clean(t):
    return re.sub(r"^[\W_]+|[\W_]+$","", " ".join(t.casefold().split()))

def score(row,output,hit_cap=False,hit_newline=False):
    from collections import Counter
    txt=clean(output);gold=clean(row["gold"])
    ta=txt.split();tb=gold.split()
    k=sum((Counter(ta)&Counter(tb)).values())
    return {"kind":row["kind"],"strict":int(txt==gold),
      "first_line":int(clean(output.split("\n",1)[0])==gold),
      "unknown":int(txt=="unknown"),"f1":2*k/(len(ta)+len(tb)) if ta and tb else 0.,
      "cap":int(hit_cap),"newline":int(hit_newline),"empty":int(not txt)}

def aggregate(rows,outs):
    if len(rows)!=len(outs):raise RuntimeError("HOLD incomplete QA generation")
    d={}
    for kind in KINDS:
        subset=[v for r,v in zip(rows,outs) if r["kind"]==kind]
        if len(subset)!=N_PAPERS:raise RuntimeError("HOLD wrong cohort size")
        d[kind]={"n":len(subset)}
        for field in ("strict","first_line","unknown","cap","newline","empty"):
            d[kind][field]=sum(v[field] for v in subset)
        d[kind]["mean_f1"]=round(sum(v["f1"] for v in subset)/len(subset),6)
    return d

if __name__=="__main__":
    rows=build()
    print("EXP045_SOURCE_SHA",SOURCE_SHA)
    print("EXP045_ELIGIBLE_PAPERS",len(candidates()))
    print("EXP045_CASES",len(rows),"topics",N_PAPERS,"cohort",receipt(rows))
