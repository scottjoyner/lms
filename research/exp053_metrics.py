#!/usr/bin/env python3
"""EXP053 analysis-only deterministic, fail-closed paired article statistics.

No model imports, inference, network, licensed text, or reviewer simulation.
This pure function is NOT an admission-verification step.
"""
import hashlib,math,random,statistics
from pathlib import Path
from collections import defaultdict
ANALYSIS_PLAN=Path(__file__).resolve().parent/"EXP053-MEASUREMENT-PLAN-2026-10-16.md"
ANALYSIS_PLAN_SHA="5239be2dd2413af98759a2ac5fdde100f2407d2de07149347f272d1e5ee0922d"
MODELS=("K2-Horizon-3.7B","Qwen3.5-4B-Base","MiniCPM-V-4.7-1B")
ARMS=("supported","withheld","decoy")
ROTATIONS=(0,1)
LETTERS="ABCD"
FIELDS={"model","case_id","article_sha256","arm","rotation",
        "context_article_sha256","checkpoint_sha256","tokenizer_sha256",
        "option_order_sha256","prompt_sha256","gold_label","foil_label",
        "gold_token_count","foil_token_count","gold_logp","foil_logp","mode"}

def hold(msg):raise ValueError("HOLD EXP053 METRICS: "+msg)
def hex64(v):return isinstance(v,str) and len(v)==64 and all(c in "0123456789abcdef" for c in v)
def rotate(s):return LETTERS[(LETTERS.index(s)+2)%4]
def mean(v):return sum(v)/len(v)
def boot(values,seed,n=2000):
    rng=random.Random(int(hashlib.sha256(seed.encode()).hexdigest(),16))
    length=len(values);means=[]
    for _ in range(n):means.append(mean([values[rng.randrange(length)] for _ in range(length)]))
    means.sort()
    return [round(means[int(.025*(n-1))],6),
            round(means[int(.975*(n-1))],6)]
def signflip(values,seed,n=5000):
    rng=random.Random(int(hashlib.sha256(seed.encode()).hexdigest(),16))
    observed=mean(values);ge=0
    for _ in range(n):
        t=mean([x if rng.getrandbits(1) else -x for x in values])
        if t>=observed-1e-12:ge+=1
    return (ge+1)/(n+1)
def holm(p):
    ordered=sorted(p,key=lambda x:p[x]);adj={};peak=0.
    for j,m in enumerate(ordered):
        peak=max(peak,min(1.,(len(ordered)-j)*p[m]))
        adj[m]=round(peak,6)
    return adj
def analyze(rows,approved,manifest):
    """Only analyze an externally authenticated human-approved set.

    Callers MUST separately validate human signatures, canonical corpus custody
    and admission before passing approved. This function cannot do that.
    """
    if not ANALYSIS_PLAN.is_file() or hashlib.sha256(ANALYSIS_PLAN.read_bytes()).hexdigest()!=ANALYSIS_PLAN_SHA:
        hold("frozen measurement plan altered or unavailable")
    if not isinstance(approved,dict) or len(approved)<40:
        hold("fewer than 40 human-reviewed case IDs")
    pub={r["case_id"]:r["article_sha256"] for r in manifest["rows"]}
    if len(pub)!=len(manifest["rows"]) or not set(approved)<=set(pub):
        hold("approved set not in sealed public cohort")
    if len(set(pub[c] for c in approved))!=len(approved):
        hold("approved cases reuse article text")
    keys=sorted(approved)
    labels={}
    for cid,ans in approved.items():
        g=ans.get("supported");f=ans.get("foil")
        if g not in LETTERS or f not in LETTERS or g==f:
            hold("invalid human-supported option or foil")
        labels[cid]=(g,f)
    next_article={cid:pub[keys[(i+1)%len(keys)]] for i,cid in enumerate(keys)}
    seen=set();margins=defaultdict(dict);model_identity={}
    label_orders=defaultdict(dict);prompts=defaultdict(set)
    if not isinstance(rows,list):hold("expected a complete list of score receipts")
    expected=len(approved)*len(MODELS)*len(ARMS)*len(ROTATIONS)
    if len(rows)!=expected:hold(f"expected {expected} rows, got {len(rows)}")
    for rec in rows:
        if not isinstance(rec,dict) or set(rec)!=FIELDS:
            hold("unexpected receipt fields (no source text may be included)")
        m=rec["model"];cid=rec["case_id"];arm=rec["arm"];rot=rec["rotation"]
        if m not in MODELS or cid not in approved or arm not in ARMS or type(rot)!=int or rot not in ROTATIONS:
            hold("unexpected model/case/arm/rotation")
        key=(m,cid,arm,rot)
        if key in seen:hold("duplicate result row")
        seen.add(key)
        if rec["mode"]!="raw":hold("chat or other model mode must be analyzed separately")
        if rec["article_sha256"]!=pub[cid]:hold("paper hash differs from public manifest")
        expected_ctx=(pub[cid] if arm=="supported" else
                      "WITHHELD" if arm=="withheld" else next_article[cid])
        if rec["context_article_sha256"]!=expected_ctx:
            hold("source arm is not deterministic supported/withheld/next-article decoy")
        for k in ("checkpoint_sha256","tokenizer_sha256","option_order_sha256","prompt_sha256"):
            if not hex64(rec[k]):hold("invalid "+k)
        ident=(rec["checkpoint_sha256"],rec["tokenizer_sha256"])
        if m in model_identity and model_identity[m]!=ident:
            hold("model weights/tokenizer receipt changes within experiment")
        model_identity[m]=ident
        good=labels[cid]
        if rot==1:good=(rotate(good[0]),rotate(good[1]))
        if (rec["gold_label"],rec["foil_label"])!=good:
            hold("model output label rotation does not match blinded human foil")
        if rec["gold_token_count"]!=1 or rec["foil_token_count"]!=1:
            hold("candidate answer-choice tokens must each be exactly one")
        x,y=rec["gold_logp"],rec["foil_logp"]
        if any(type(v) not in (int,float) or not math.isfinite(v) or v>0 for v in (x,y)):
            hold("invalid next-token log probabilities")
        orkey=(cid,rot)
        if orkey in label_orders and label_orders[orkey]!=rec["option_order_sha256"]:
            hold("option permutation changes between source arms")
        label_orders[orkey]=rec["option_order_sha256"]
        pkey=(m,cid,rot)
        if rec["prompt_sha256"] in prompts[pkey]:hold("source arms share identical prompt")
        prompts[pkey].add(rec["prompt_sha256"])
        margins[(m,cid)][(arm,rot)]=x-y
    if len(seen)!=expected or len(model_identity)!=len(MODELS):
        hold("incomplete or repeated case-model-source grid")
    values={};report={}
    for m in MODELS:
        trio={}
        for cid in keys:
            case=margins[m,cid]
            if len(case)!=6:hold("missing paired conditions")
            avg={arm:mean([case[arm,r] for r in ROTATIONS]) for arm in ARMS}
            trio[cid]={"D":avg["supported"]-avg["withheld"],
                       "N":avg["decoy"]-avg["withheld"],
                       "S":avg["supported"]-avg["decoy"]}
        values[m]=trio
        metric={}
        for typ in ("D","N","S"):
            z=[trio[c][typ] for c in keys]
            metric[typ]={"mean":round(mean(z),6),"median":round(statistics.median(z),6),
                         "positive":sum(1 for v in z if v>0),"n":len(z),
                         "ci95_article_bootstrap":boot(z,"EXP053-metrics-bootstrap-v1|"+m+"|"+typ)}
        metric["p_D_signflip_unadjusted"]=round(signflip(
            [trio[c]["D"] for c in keys],"EXP053-metrics-signflip-v1|"+m),6)
        report[m]=metric
    corrected=holm({m:report[m]["p_D_signflip_unadjusted"] for m in MODELS})
    for m in MODELS:report[m]["p_D_holm_3models"]=corrected[m]
    return {"schema":"exp053-model-free-analysis-v1","n_approved_articles":len(keys),
            "expected_and_observed_score_rows":expected,"models":report,
            "caution":"Exploratory evidence sensitivity, not proof of grounding, unseen data, or lineage."}
