"""EXP049: same-evidence human gold versus length-matched foil, 8 frozen QASPER papers."""
import hashlib,json,re
from transformers import AutoTokenizer
from k2_interpolation_pilot import MID
from k2_qasper_decode_fixture import cohort,receipt,prompt
ORIGINAL_SHA="0f9c4fad7ba905a3b1283911ea50b77f4470b710183509425e24bd14328f69c5"
MISSING="The research-paper excerpt was withheld."
MODES=("default","yarn")
ARMS=("original","withheld")
def digest(s):return hashlib.sha256(s.encode("utf8")).hexdigest()
def lexical(t):return set(re.findall(r"[a-z]{3,}",t.casefold()))
def make():
    allrows=cohort()
    if len(allrows)!=16 or receipt(allrows)!=ORIGINAL_SHA:
        raise RuntimeError("HOLD changed original QASPER sources")
    tok=AutoTokenizer.from_pretrained(str(MID),local_files_only=True,trust_remote_code=False)
    base=[r for r in allrows if r["kind"]=="supported"]
    if len(base)!=8:raise RuntimeError("HOLD must have eight supported sources")
    rows=[]
    for row in base:
        answer=row["gold"];parts=list(re.finditer(r"\b[\w-]+\b",row["context"]))
        n=len(tok.encode(" "+answer,add_special_tokens=False))
        if not 1<=n<=32:raise RuntimeError("HOLD invalid gold token count")
        pool=[]
        for i in range(len(parts)):
            for j in range(i+1,min(len(parts),i+29)+1):
                cand=row["context"][parts[i].start():parts[j-1].end()]
                if len(cand)<3 or cand.casefold()==answer.casefold():continue
                if lexical(cand)&lexical(answer):continue
                if len(tok.encode(" "+cand,add_special_tokens=False))!=n:continue
                pool.append((abs(len(cand)-len(answer)),
                             abs((j-i)-len(answer.split())),digest(cand),cand))
        if not pool:raise RuntimeError("HOLD no exact-token-length same-evidence foil")
        foil=sorted(pool)[0][-1]
        if not foil in row["context"] or answer not in row["context"]:
            raise RuntimeError("HOLD gold or foil not in shared source")
        if lexical(foil)&lexical(answer):
            raise RuntimeError("HOLD foil overlaps gold content words")
        for arm in ARMS:
            change=dict(row)
            if arm=="withheld":change["context"]=MISSING
            rows.append({"topic":row["topic"],"arm":arm,"row":change,
                         "candidate_gold":" "+answer,"candidate_foil":" "+foil,
                         "n":n})
    if len(rows)!=16 or len({r["topic"] for r in rows})!=8:
        raise RuntimeError("HOLD cohort shape")
    return rows
def source_digest(items=None):
    items=make() if items is None else items
    data=[{"topic":v["topic"],"arm":v["arm"],"prompt":prompt(v["row"]),
           "gold":v["candidate_gold"],"foil":v["candidate_foil"],"n":v["n"]}
          for v in items]
    return digest(json.dumps(data,ensure_ascii=False,sort_keys=True,separators=(",",":")))
if __name__=="__main__":
    r=make()
    print("EXP049_CONTEXTS",len(r),"GOLD_FOIL_SAME_TOKEN_LENGTH",
          all(r["n"]>=1 for r in r),"COHORT_SHA",source_digest(r))
