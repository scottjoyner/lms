#!/usr/bin/env python3
"""EXP046: out-of-EXP045 paper retrieval (BM25 top1/top3) vs oracle evidence.

Retriever has full paper paragraphs in local source, generator sees only
selected snippets. All source selection and ranking are model-independent.
"""
import collections,hashlib,json,math,re
from k2_qasper_evidence_fixture import (
 dataset,candidates,sha,lexical,SOURCE_SHA,STOP)
from k2_qasper_evidence_fixture import score as evaluate_score

PAPERS=12
KINDS=("oracle_supported","bm25_answer_top1","bm25_answer_top3",
       "bm25_unanswer_top1","bm25_unanswer_top3")
SEED="EXP046-QASPER-full-index-BM25-next12-20261015"
MAX_LEN=900
PARA_MIN=100

def toks(x):
    return [v for v in re.findall(r"[a-z0-9]{3,}",x.casefold()) if v not in STOP]

def snippets(paper):
    indexed=[];seen=set()
    for sec in paper["full_text"]:
        for raw in sec["paragraphs"]:
            if not isinstance(raw,str):continue
            txt=raw.strip()
            if not txt:continue
            # Fixed non-overlapping hard windows so any full paper is indexable
            # without changing the tokenizer budget or inspecting gold strings.
            for i in range(0,len(txt),MAX_LEN):
                snip=txt[i:i+MAX_LEN].strip()
                if len(snip)<PARA_MIN or snip in seen:continue
                seen.add(snip);indexed.append(snip)
    if len(indexed)<3:raise RuntimeError("HOLD paper has <3 indexing windows")
    return indexed

def rank(query,parts):
    # Fixed conventional BM25 with k1=1.2 b=.75, deterministic original-index
    # tie-break. No trained/specialized reranker, no model-generated queries.
    terms=toks(query)
    corpus=[toks(p) for p in parts]
    N=len(corpus)
    avg=sum(len(v) for v in corpus)/N
    df=collections.Counter(token for row in corpus for token in set(row))
    qset=set(terms)
    results=[]
    for i,doc in enumerate(corpus):
        count=collections.Counter(doc)
        score=0.
        for token in qset:
            if token not in count:continue
            idf=math.log(1+(N-df[token]+.5)/(df[token]+.5))
            tf=count[token]
            denom=tf+1.2*(1-.75+.75*len(doc)/avg)
            score+=idf*(tf*2.2)/denom
        results.append((-score,i))
    return [i for _,i in sorted(results)]

def examples():
    all_candidates=candidates()
    if len(all_candidates)<24:raise RuntimeError("HOLD insufficient disjoint QASPER papers")
    # Original 12 EXP045 source papers excluded by using ranked next 12.
    previous={pid for _,pid,*_ in all_candidates[:12]}
    later=all_candidates[12:24]
    if any(pid in previous for _,pid,*_ in later):
        raise RuntimeError("HOLD EXP045 topic reuse")
    rows=[];retrieval=[]
    for topic,(_,pid,p,answerable,unknown,_,_) in enumerate(later):
        qa,gold,human_ev=answerable
        question=qa["question"].strip()
        unq=unknown["question"].strip()
        parts=snippets(p)
        ranked_a=rank(question,parts)
        ranked_u=rank(unq,parts)
        top1_a=parts[ranked_a[0]]
        top3_a="\n\n".join(parts[j] for j in ranked_a[:3])
        top1_u=parts[ranked_u[0]]
        top3_u="\n\n".join(parts[j] for j in ranked_u[:3])
        def row(kind,orig,context,ref):
            return {"topic":topic,"pid":pid,"title":p["title"].strip(),
                    "qid":orig["question_id"],"kind":kind,
                    "question":orig["question"].strip(),"gold":ref,"context":context}
        rows.extend((
            row("oracle_supported",qa,human_ev,gold),
            row("bm25_answer_top1",qa,top1_a,gold),
            row("bm25_answer_top3",qa,top3_a,gold),
            row("bm25_unanswer_top1",unknown,top1_u,"UNKNOWN"),
            row("bm25_unanswer_top3",unknown,top3_u,"UNKNOWN")))
        retrieval.append({
         "answer_gold_top1":int(gold.casefold() in top1_a.casefold()),
         "answer_gold_top3":int(gold.casefold() in top3_a.casefold()),
         "answer_gold_any_index":int(any(gold.casefold() in txt.casefold() for txt in parts)),
         "index_windows":len(parts),
         "answer_top1_overlap":len(lexical(question)&lexical(top1_a)),
         "unanswer_top1_overlap":len(lexical(unq)&lexical(top1_u)),
         "answer_top3_context_chars":len(top3_a),
         "unanswer_top3_context_chars":len(top3_u)})
    if len(rows)!=60 or len({r["pid"] for r in rows})!=12:raise RuntimeError("HOLD 12x5 input shape")
    for r in rows:
        if r["kind"]=="oracle_supported" and r["gold"] not in r["context"]:
            raise RuntimeError("HOLD human gold absent from oracle evidence")
        if r["kind"].startswith("bm25_unanswer") and r["gold"]!="UNKNOWN":
            raise RuntimeError("HOLD human unanswerable gold changed")
    return rows,retrieval

def prompt(r):
    return ("Using ONLY the provided research-paper evidence windows, "
       "give the shortest exact answer phrase supported by the excerpts, "
       "or exactly UNKNOWN if the supplied evidence does not establish an answer. "
       "Write one line and stop.\n"
       "Paper: "+r["title"]+"\nEvidence:\n"+r["context"]+
       "\nQuestion: "+r["question"]+"\nAnswer:")

def receipt(rows=None,retrieval=None):
    if rows is None or retrieval is None:
        rows,retrieval=examples()
    content=[{"kind":r["kind"],"topic":r["topic"],"qid":r["qid"],
              "gold":r["gold"],"prompt":prompt(r)} for r in rows]
    combined={"prompts":content,"retrieval":retrieval}
    return sha(json.dumps(combined,sort_keys=True,ensure_ascii=False,separators=(",",":")))

def score(row,output,hit_cap=False,hit_newline=False):
    return evaluate_score(row,output,hit_cap=hit_cap,hit_newline=hit_newline)

def aggregate(rows,outputs):
    if len(rows)!=len(outputs):raise RuntimeError("HOLD incomplete model outputs")
    result={}
    for kind in KINDS:
        selected=[v for r,v in zip(rows,outputs) if r["kind"]==kind]
        if len(selected)!=PAPERS:raise RuntimeError("HOLD per-case count")
        result[kind]={"n":PAPERS}
        for field in ("strict","first_line","unknown","newline","cap","empty"):
            result[kind][field]=sum(v[field] for v in selected)
        result[kind]["mean_f1"]=round(sum(v["f1"] for v in selected)/PAPERS,6)
    return result

def retrieval_audit(stats):
    if len(stats)!=PAPERS:raise RuntimeError("HOLD retrieval stats")
    return {
      "papers":len(stats),
      "answer_gold_top1":sum(x["answer_gold_top1"] for x in stats),
      "answer_gold_top3":sum(x["answer_gold_top3"] for x in stats),
      "answer_gold_any_index":sum(x["answer_gold_any_index"] for x in stats),
      "index_windows_min":min(x["index_windows"] for x in stats),
      "index_windows_max":max(x["index_windows"] for x in stats),
      "mean_answer_top1_overlap":round(sum(x["answer_top1_overlap"] for x in stats)/PAPERS,3),
      "mean_unanswer_top1_overlap":round(sum(x["unanswer_top1_overlap"] for x in stats)/PAPERS,3)
    }

if __name__=="__main__":
    rows,stats=examples()
    print("EXP046_ROWS",len(rows),"topics",PAPERS)
    print("EXP046_COHORT_SHA",receipt(rows,stats))
    print("EXP046_RETRIEVAL_AGGREGATES",retrieval_audit(stats))
