#!/usr/bin/env python3
"""Source-only exact/20-word overlap audit (NOT pretraining contamination proof)."""
import hashlib,json,re,unicodedata
from pathlib import Path
from exp053_race_review import MANIFEST,MANIFEST_SHA,SOURCE_SHA,digest,filehash,deny
PRIOR=Path("/home/scott/git/wt-lms-k2-interpolation-pilot-20261015/research/data")
PRIOR_FILES={"qasper":("qasper-dev-v0.3.json",
                         "2ae7ee62a65b1c4225791c70de80c2aad4e8998cf1fd4f09a53103db4f21af93"),
             "quac":("quac-val-v0.2.json",
                     "09e622916280ba04c9352acb1bc5bbe80f11a2598f6f34e934c51d9e6570f378")}
PRIVATE=Path("/media/scott/SSD_4TB/exp053-private-20261016/packet.json")
def words(text):return re.findall(r"\w+",unicodedata.normalize("NFKC",text).casefold())
def shingles(w,n=20):return zip(*(w[i:] for i in range(n))) if len(w)>=n else ()
def sources(kind,payload):
    if kind=="quac":
        for article in payload["data"]:
            for paragraph in article["paragraphs"]:
                yield paragraph["context"]
    else:
        for paper in payload.values():
            for section in paper["full_text"]:
                for paragraph in section["paragraphs"]:
                    if isinstance(paragraph,str):yield paragraph
def run():
    if filehash(MANIFEST)!=MANIFEST_SHA:deny("public cohort hash mismatch")
    manifest=json.loads(MANIFEST.read_text())
    packet=json.loads(PRIVATE.read_text())
    if digest(packet)!=manifest["packet_sha256"]:deny("private packet hash mismatch")
    assert len(packet["cases"])==64
    test={x["case_id"]:words(x["article"]) for x in packet["cases"]}
    original_shas={x["case_id"]:hashlib.sha256(x["article"].encode()).hexdigest()
                   for x in packet["cases"]}
    if len(set(original_shas.values()))!=64:deny("duplicate source articles")
    for row in manifest["rows"]:
        if original_shas[row["case_id"]]!=row["article_sha256"]:
            deny("public article hash mismatch")
    index={}
    for cid,w in test.items():
        for g in set(shingles(w)):
            index.setdefault(g,set()).add(cid)
    counts={}
    for name,(filename,expect) in PRIOR_FILES.items():
        path=PRIOR/filename
        if not path.is_file() or filehash(path)!=expect:deny(name+" checksum drift")
        payload=json.loads(path.read_text())
        hits=set();total=0;exact=0
        canon_test={tuple(w) for w in test.values()}
        for passage in sources(name,payload):
            total+=1;pw=words(passage)
            if tuple(pw) in canon_test:exact+=1
            for gram in set(shingles(pw)):
                hits.update(index.get(gram,()))
        counts[name]={"full_passages_examined":total,
                      "exact_full_article_matches":exact,
                      "race_cases_with_at_least_one_shared_20_word_span":len(hits),
                      "shared_case_id_hashes":sorted(hashlib.sha256(x.encode()).hexdigest() for x in hits),
                      "source_sha256":expect}
    return {"schema":"exp053-source-overlap-v1","source_sha256":SOURCE_SHA,
            "manifest_sha256":MANIFEST_SHA,"race_articles":len(test),
            "distinct_race_article_text":len(set(original_shas.values())),
            "shared_20word_tests":counts,
            "squad_audited":False,
            "pretraining_overlap_ruled_out":False,
            "limitation":"Exact/20-token partial overlap against local public QASPER/QuAC copies only; not a model-training-contamination test."}
if __name__=="__main__":
    print(json.dumps(run(),indent=2,sort_keys=True))
