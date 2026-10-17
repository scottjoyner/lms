#!/usr/bin/env python3
"""EXP053 source-locked RACE blinded packet and signed review gate. No model access."""
import argparse,base64,hashlib,json,os
from pathlib import Path
from collections import defaultdict
SOURCE=Path("/media/scott/SSD_4TB/hf-cache/huggingface/hub/datasets--ehovy--race/snapshots/2fec9fd81f1dc971569a9b729c43f2f0e6436637/high/test-00000-of-00001.parquet")
SOURCE_SHA="97ef697f4c81e804eb43e7a667a6ac7bb06c11a99cfa537997ae5e2f6c5db2b7"
PROTOCOL=Path(__file__).resolve().parent/"EXP053-PROTOCOL-2026-10-16.md"
PROTOCOL_SHA="c37b25b9cd5d08015a05bd1e5791b9e0c8d9d6367cf9599149106b6800c681d8"
MANIFEST=Path(__file__).resolve().parent/"EXP053-COHORT-MANIFEST-2026-10-16.json"
MANIFEST_SHA="43e8309ac1c8ebdb28ae1c478254d43313449e0a44588c301f0a984a240d92b9"
SEED="EXP053-RACE-HIGH-TEST-v1"
MIN_REVIEWED=40
TARGET=64
LETTERS="ABCD"

def canonical(x):return json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()
def digest(x):return hashlib.sha256(x if isinstance(x,bytes) else canonical(x)).hexdigest()
def filehash(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()
def deny(msg):raise ValueError("HOLD EXP053: "+msg)
def corpus():
    if not SOURCE.is_file() or filehash(SOURCE)!=SOURCE_SHA:deny("source authority mismatch")
    if filehash(PROTOCOL)!=PROTOCOL_SHA:deny("preregistered protocol drift")
    import pyarrow.parquet as pq
    items=pq.read_table(SOURCE).to_pylist()
    if len(items)<2500:deny("insufficient test corpus")
    return items
def cohort(rows):
    grouped=defaultdict(list);article_hash={}
    for r in rows:
        aid=r["example_id"];article=r["article"].strip()
        if not isinstance(aid,str) or not aid or not 250<=len(article)<=3500:continue
        if aid in article_hash and article_hash[aid]!=digest(article.encode()):
            deny("article id maps to multiple sources")
        article_hash[aid]=digest(article.encode())
        opts=[x.strip() for x in r["options"]]
        question=r["question"].strip()
        if not (15<=len(question)<=230 and len(opts)==4 and
                len(set(x.casefold() for x in opts))==4 and
                all(1<=len(x)<=160 for x in opts) and r["answer"] in LETTERS):
            continue
        grouped[aid].append({"article":article,"question":question,"options":opts,
                             "key":r["answer"],"article_id":aid})
    if len(grouped)<TARGET:deny("fewer than 64 eligible distinct articles")
    selection=[]
    for aid in sorted(grouped,key=lambda x:digest((SEED+"|"+x).encode()))[:TARGET]:
        candidates=grouped[aid]
        item=min(candidates,key=lambda x:digest((SEED+"|"+x["question"]+"|"+
                                "|".join(x["options"])).encode()))
        item=dict(item);item["case_id"]=digest((SEED+"|"+aid+"|"+item["question"]).encode())[:20]
        perm=sorted(range(4),key=lambda i:digest((SEED+"|rotate|"+item["case_id"]+"|"+str(i)).encode()))
        item["blinded_options"]=[item["options"][i] for i in perm]
        item["blinded_key"]=LETTERS[perm.index(LETTERS.index(item["key"]))]
        selection.append(item)
    if len({x["case_id"] for x in selection})!=TARGET:deny("duplicate source identity")
    if len({digest(x["article"].encode()) for x in selection})!=TARGET:
        deny("distinct article IDs reuse the same passage")
    return selection
def build_packet(items):
    packet={"schema":"exp053-packet-v1","seed":SEED,
            "source_sha256":SOURCE_SHA,"protocol_sha256":PROTOCOL_SHA,
            "license":"RACE: non-commercial research only; do not redistribute",
            "cases":[{"case_id":x["case_id"],"article":x["article"],"question":x["question"],
                     "options":dict(zip(LETTERS,x["blinded_options"]))} for x in items]}
    return packet
def public_manifest(items,packet):
    source=[{"case_id":x["case_id"],
             "article_sha256":digest(x["article"].encode()),
             "question_sha256":digest(x["question"].encode()),
             "options_sha256":digest(canonical(x["blinded_options"])),
             "article_chars":len(x["article"])}
            for x in items]
    sealed={x["case_id"]:x["blinded_key"] for x in items}
    return {"schema":"exp053-public-manifest-v1",
            "source":"ehovy/race high/test rev 2fec9fd81f1dc971569a9b729c43f2f0e6436637",
            "source_sha256":SOURCE_SHA,"protocol_sha256":PROTOCOL_SHA,
            "packet_sha256":digest(packet),"sealed_answer_key_sha256":digest(sealed),
            "candidate_articles":len(items),"human_reviewed_articles":0,
            "minimum_human_reviewed":MIN_REVIEWED,"evaluation_admitted":False,
            "copyright":"no licensed RACE text reproduced in repository","rows":source}
def save_private(path,obj):
    with open(path,"xb") as f:
        os.fchmod(f.fileno(),0o600)
        f.write(canonical(obj)+b"\n")
def build(private_dir,public):
    if public.exists() or not public.parent.is_dir():deny("public output exists / parent absent")
    if private_dir.exists():deny("private review directory already exists")
    items=cohort(corpus());packet=build_packet(items)
    manifest=public_manifest(items,packet)
    private_dir.mkdir(mode=0o700)
    save_private(private_dir/"packet.json",packet)
    save_private(private_dir/"sealed_exam_key.json",{
        "packet_sha256":digest(packet),"answer_key":{i["case_id"]:i["blinded_key"] for i in items}})
    for slot in ("A","B"):
        save_private(private_dir/("reviewer_"+slot+".json"),{
            "schema":"exp053-review-v1","packet_sha256":digest(packet),
            "reviewer_id":"","role":"independent","attestation":"","decisions":[
               {"case_id":i["case_id"],"decision":"unreviewed",
                "supported_option":"","foil_option":"","support_quote":"",
                "foil_reason":"","reject_reason":""} for i in items],"signature":""})
    save_private(public,manifest)
    print("EXP053_CANDIDATE_PACKET_READY",len(items),"articles; human_reviews=0;",
          "FULL_INFERENCE_HOLD=True; manifest",digest(manifest))
    return manifest
def verify_signed(review,trust):
    if review.get("schema")!="exp053-review-v1":deny("unexpected review schema")
    rid=review.get("reviewer_id")
    if not isinstance(rid,str) or rid not in trust:deny("untrusted reviewer key")
    if review.get("attestation")!="I independently reviewed these source passages and responses; no model predictions or other reviewer labels were provided to me.":
        deny("missing independent human review attestation")
    if not isinstance(trust,dict) or len(trust)<2 or len(set(trust.values()))!=len(trust):
        deny("trust registry requires distinct signer public keys")
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    try:
        pub=Ed25519PublicKey.from_public_bytes(base64.b64decode(trust[rid],validate=True))
        signature=bytes.fromhex(review["signature"])
        unsigned={k:v for k,v in review.items() if k!="signature"}
        pub.verify(signature,canonical(unsigned))
    except Exception as exc:
        deny("signature validation failed: "+type(exc).__name__)
    return rid
def check_decision(dec,case):
    status=dec.get("decision")
    if status=="reject":
        if len(dec.get("reject_reason","").strip())<12:deny("unreasoned reject")
        return status
    if status!="accept":deny("unreviewed or invalid decision")
    a=dec.get("supported_option");f=dec.get("foil_option")
    if a not in LETTERS or f not in LETTERS or a==f:
        deny("invalid chosen supported/foil options")
    if len(dec.get("support_quote","").strip())<20 or dec["support_quote"].strip() not in case["article"]:
        deny("source quote absent or too short")
    if len(dec.get("foil_reason","").strip())<15:
        deny("foil reason missing")
    return status
def decisions_for(review,packet,required_ids=None):
    if review.get("packet_sha256")!=digest(packet):deny("packet mismatch")
    if review.get("role") not in ("independent","adjudicator"):deny("invalid signer role")
    lookup={c["case_id"]:c for c in packet["cases"]}
    ds=review.get("decisions",[])
    expected=set(lookup) if required_ids is None else set(required_ids)
    if not isinstance(ds,list) or len(ds)!=len(expected):deny("incomplete review form")
    out={}
    for d in ds:
        cid=d.get("case_id")
        if cid in out or cid not in lookup:deny("duplicate/foreign case")
        check_decision(d,lookup[cid]);out[cid]=d
    if set(out)!=expected:deny("unexpected decision scope")
    return out
def gate(private_dir,keys_path,output,adjudicator_file=None):
    if output.exists():deny("gate output exists")
    if not SOURCE.is_file() or filehash(SOURCE)!=SOURCE_SHA:deny("source authority changed")
    if not PROTOCOL.is_file() or filehash(PROTOCOL)!=PROTOCOL_SHA:deny("protocol drift")
    if not MANIFEST.is_file() or filehash(MANIFEST)!=MANIFEST_SHA:
        deny("immutable public manifest is absent or modified")
    manifest=json.loads(MANIFEST.read_text())
    packet=json.loads((private_dir/"packet.json").read_text())
    seal=json.loads((private_dir/"sealed_exam_key.json").read_text())
    if packet.get("source_sha256")!=SOURCE_SHA or packet.get("protocol_sha256")!=PROTOCOL_SHA:
        deny("packet custody mismatch")
    if digest(packet)!=manifest["packet_sha256"]:deny("packet differs from manifest")
    if (seal["packet_sha256"]!=digest(packet) or
        digest(seal["answer_key"])!=manifest["sealed_answer_key_sha256"]):
        deny("answer key differs from committed manifest")
    trust=json.loads(keys_path.read_text());observations=[];signer_ids=[]
    for slot in ("A","B"):
        review=json.loads((private_dir/("reviewer_"+slot+".json")).read_text())
        signer_ids.append(verify_signed(review,trust))
        if review["role"]!="independent":deny("reviewer role")
        observations.append(decisions_for(review,packet))
    if signer_ids[0]==signer_ids[1] or trust[signer_ids[0]]==trust[signer_ids[1]]:
        deny("two reviewers must have independent signer keys")
    accepted={};rejected=0;unresolved=[]
    for cid,key in seal["answer_key"].items():
        a,b=[rv[cid] for rv in observations]
        if a["decision"]=="reject" and b["decision"]=="reject":
            rejected+=1;continue
        if (a["decision"]=="accept" and b["decision"]=="accept" and
            a["supported_option"]==b["supported_option"]==key and
            a["foil_option"]==b["foil_option"]):
            accepted[cid]={"supported":key,"foil":a["foil_option"]}
        else:unresolved.append(cid)
    if unresolved:
        if adjudicator_file is None:
            deny(str(len(unresolved))+" disputed items require independent third review")
        third=json.loads(adjudicator_file.read_text())
        third_id=verify_signed(third,trust)
        if third_id in signer_ids or third["role"]!="adjudicator":
            deny("invalid independent adjudicator")
        ruling=decisions_for(third,packet,required_ids=unresolved)
        if third_id in signer_ids or trust[third_id] in [trust[x] for x in signer_ids]:
            deny("adjudicator key must be independent")
        signer_ids.append(third_id)
        for cid in unresolved:
            v=ruling[cid]
            if v["decision"]=="reject":
                rejected+=1;continue
            if v["supported_option"]!=seal["answer_key"][cid]:
                deny("third ruling does not match sealed human exam key")
            accepted[cid]={"supported":v["supported_option"],"foil":v["foil_option"]}
    if len(accepted)<MIN_REVIEWED:
        deny("only "+str(len(accepted))+" independently adjudicated accepted articles")
    save_private(output,{"schema":"exp053-admitted-v1","packet_sha256":digest(packet),
        "reviewers":signer_ids,"accepted":accepted,"rejected":rejected,
        "human_approval":True,"simulation":False})
    return len(accepted)
def main():
    ap=argparse.ArgumentParser()
    sub=ap.add_subparsers(dest="cmd",required=True)
    b=sub.add_parser("build")
    b.add_argument("--private-dir",type=Path,required=True)
    b.add_argument("--public-manifest",type=Path,required=True)
    g=sub.add_parser("gate")
    g.add_argument("--private-dir",type=Path,required=True)
    g.add_argument("--trust-roots",type=Path,required=True)
    g.add_argument("--out",type=Path,required=True)
    g.add_argument("--adjudicator-file",type=Path)
    args=ap.parse_args()
    if args.cmd=="build":
        build(args.private_dir,args.public_manifest)
    else:
        n=gate(args.private_dir,args.trust_roots,args.out,args.adjudicator_file)
        print("EXP053_HUMAN_ADJUDICATION_VERIFIED",n)
if __name__=="__main__":main()
