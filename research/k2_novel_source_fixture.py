#!/usr/bin/env python3
"""EXP041 — never-published seeded synthetic source QA, zero model import.

These materials are synthetic originals, not naturalistic SQuAD or an
independently curated population-level task. They are generated AFTER
the pinned K2 model checkpoints; no training-unseen guarantee is made.
"""
import hashlib
import json
import re

SEED = "EXP041-20261015-pinned-synthetic-ledger-v1"
ARTIFACTS = ("bronze cylinder", "cobalt spool", "silver pendant",
             "amber vial", "copper capsule", "violet ledger",
             "jade compass", "pearl canister")
RECORDS=8
MAX_NEW=16
MODES=("default","yarn")
CASES=("answerable","unanswerable","redacted","swapped")

def digest(text):
    return hashlib.sha256(text.encode("utf8")).hexdigest()

def nonce(index,part):
    # Pronounceable generated proper nouns; not sourced from a public QA corpus.
    chunks=("brel","kav","orin","tav","lume","sair","drel","vorn",
            "nemi","fex","yaro","mav","zun","vale","tori","vek")
    h=hashlib.sha256(f"{SEED}:person:{index}:{part}".encode()).digest()
    return "".join(chunks[h[j]%len(chunks)] for j in range(2)).title()

def build():
    rows=[]
    for i,artifact in enumerate(ARTIFACTS):
        carrier=nonce(i,"carrier");clerk=nonce(i,"clerk")
        if carrier==clerk:raise RuntimeError("HOLD ambiguous agent names")
        other=ARTIFACTS[(i+3)%RECORDS]
        location=("north cabinet","west shelf","east bay","lower locker",
                  "upper crate","south archive","central bench","rear vault")[i]
        passage=(f"Ledger code {digest(SEED+str(i))[:12]}: {carrier} carried "
                 f"the {artifact} into the {location}. "
                 f"{clerk} inspected the {other} near the {location}. "
                 f"The shift ended with a folded map beside the {location}.")
        rows.append({"topic":i,"source":passage,"answer":carrier,
            "answer_q":f"Who carried the {artifact} into the {location}?",
            "unknown_q":f"Who repaired the folded map beside the {location}?",
            "artifact":artifact})
    # All topic groups are distinct and generated, not publicly sourced.
    if len(set(r["answer"] for r in rows))!=8 or len(set(r["source"] for r in rows))!=8:
        raise RuntimeError("HOLD duplicated topic/person")
    cases=[]
    for r in rows:
        i=r["topic"]
        cases.extend((
           {"topic":i,"kind":"answerable","source":r["source"],
            "question":r["answer_q"],"gold":r["answer"]},
           {"topic":i,"kind":"unanswerable","source":r["source"],
            "question":r["unknown_q"],"gold":"UNKNOWN"}
        ))
        if i in (0,2,4,6):
            cases.append({"topic":i,"kind":"redacted",
                "source":"The original registry passage is withheld.",
                "question":r["answer_q"],"gold":"UNKNOWN"})
            cases.append({"topic":i,"kind":"swapped",
                "source":rows[(i+1)%RECORDS]["source"],
                "question":r["answer_q"],"gold":"UNKNOWN"})
    if len(cases)!=24 or {k:sum(c["kind"]==k for c in cases) for k in CASES}!={
            "answerable":8,"unanswerable":8,"redacted":4,"swapped":4}:
        raise RuntimeError("HOLD source fixture shape")
    for c in cases:
        if c["kind"]=="answerable":
            if c["gold"] not in c["source"]:raise RuntimeError("HOLD gold absent")
        elif c["kind"]=="swapped":
            # Must be no accidental evidence for the asked action in the wrong source.
            thing=ARTIFACTS[c["topic"]]
            if thing in c["source"]:raise RuntimeError("HOLD swapped artifact leakage")
        elif c["kind"]=="redacted":
            if "carried" in c["source"]:raise RuntimeError("HOLD redacted leakage")
        elif c["kind"]=="unanswerable":
            if "repaired" in c["source"]:raise RuntimeError("HOLD impossible action leakage")
    return cases

def prompt(row):
    return ("Answer the question using ONLY the passage. "
            "Give one person name copied from the passage, or exactly UNKNOWN "
            "if the passage does not establish the answer. No explanation.\n"
            f"Passage:\n{row['source']}\n"
            f"Question: {row['question']}\nAnswer:")

def corpus_receipt(cases=None):
    if cases is None:cases=build()
    return digest(json.dumps([{"topic":c["topic"],"kind":c["kind"],
      "source":c["source"],"question":c["question"],"gold":c["gold"],
      "prompt":prompt(c)} for c in cases],sort_keys=True,separators=(",",":"),
      ensure_ascii=False))

def normalize(text):
    text=" ".join(text.casefold().strip().split())
    return re.sub(r'^[\s\\.\\,:;!"\\\']+|[\s\\.\\,:;!"\\\']+$',"",text)

def score(row,completion,hitcap=False):
    clean=normalize(completion)
    gold=normalize(row["gold"])
    strict=int(clean==gold)
    first=normalize(completion.splitlines()[0]) if completion.splitlines() else ""
    line_exact=int(first==gold)
    # Lenient exact whole-word inclusion, not factual compliance.
    contains=int(bool(re.search(r"(?<!\w)"+re.escape(gold)+r"(?!\w)",clean)))
    return {"kind":row["kind"],"strict":strict,"first_line":line_exact,
      "contains_gold":contains,"said_unknown":int(clean=="unknown"),
      "empty":int(not clean),"hit_cap":int(hitcap)}

def score_group(rows,scored):
    if len(rows)!=len(scored):raise RuntimeError("HOLD score item count")
    out={}
    for kind in CASES:
        indices=[i for i,c in enumerate(rows) if c["kind"]==kind]
        if len(indices)!={"answerable":8,"unanswerable":8,"redacted":4,"swapped":4}[kind]:
            raise RuntimeError("HOLD unexpected cohort count")
        sub=[scored[i] for i in indices]
        if any(r["kind"]!=kind for r in sub):raise RuntimeError("HOLD score order drift")
        out[kind]={"n":len(sub)}
        for key in ("strict","first_line","contains_gold","said_unknown","empty","hit_cap"):
            out[kind][key]=sum(x[key] for x in sub)
    return out

if __name__=="__main__":
    cases=build()
    print("EXP041_SYNTHETIC_FIXTURE_ONLY",len(cases),"topics",RECORDS,
          "digest",corpus_receipt(cases))
    print("No model weights loaded or examples emitted.")
