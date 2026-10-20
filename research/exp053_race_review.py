#!/usr/bin/env python3
"""EXP053 source-locked RACE blinded packet and signed review gate. No model access."""
import argparse
import base64
import hashlib
import json
import math
import os
import stat
from pathlib import Path
from collections import defaultdict
SOURCE=Path("/media/scott/SSD_4TB/hf-cache/huggingface/hub/datasets--ehovy--race/snapshots/2fec9fd81f1dc971569a9b729c43f2f0e6436637/high/test-00000-of-00001.parquet")
SOURCE_SHA="97ef697f4c81e804eb43e7a667a6ac7bb06c11a99cfa537997ae5e2f6c5db2b7"
PROTOCOL=Path(__file__).resolve().parent/"EXP053-PROTOCOL-2026-10-16.md"
PROTOCOL_SHA="c37b25b9cd5d08015a05bd1e5791b9e0c8d9d6367cf9599149106b6800c681d8"
MANIFEST=Path(__file__).resolve().parent/"EXP053-COHORT-MANIFEST-2026-10-16.json"
MANIFEST_SHA="43e8309ac1c8ebdb28ae1c478254d43313449e0a44588c301f0a984a240d92b9"
# Scientific trust bootstrap: owner must independently verify distinct actual
# human identities/key possession and pin the exact public-key roster digest.
# Remains unset until owner approval. Programmatic signers are not human peers.
APPROVED_REVIEWER_ROSTER_SHA256=""
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

PRIVATE_JSON_LIMIT = 8_000_000
PUBLIC_MANIFEST_LIMIT = 1_000_000


def _unique_private_fields(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            deny("ambiguous duplicate field in private admission JSON")
        value[key] = item
    return value


def _invalid_private_constant(_value):
    deny("invalid numeric constant in private admission JSON")


def _finite_private_float(value):
    parsed = float(value)
    if not math.isfinite(parsed):
        deny("non-finite number in private admission JSON")
    return parsed


def _read_private_admission_json(path):
    """Read stable, source-protected owner-private JSON without symlink walks.

    No actual human record or RACE source is accessed by unit tests. The
    independently pinned roster must already pass before admission uses this.
    """
    from exp053_reviewer_roster import _open_no_symlink_ancestors
    if not all(getattr(os, name, 0) for name in
               ("O_PATH", "O_DIRECTORY", "O_NOFOLLOW", "O_NONBLOCK")):
        deny("secure private admission file access unavailable")
    absolute = Path(os.path.abspath(os.fspath(path)))
    try:
        root = Path(__file__).resolve().parents[1]
        intended = absolute.resolve(strict=True)
        if intended.is_relative_to(root):
            deny("private admission input inside public source checkout")
        fd = _open_no_symlink_ancestors(absolute)
        try:
            before = os.fstat(fd)
            if (not stat.S_ISREG(before.st_mode) or before.st_uid != os.geteuid() or
                    before.st_nlink != 1 or before.st_mode & 0o077 or
                    not 1 <= before.st_size <= PRIVATE_JSON_LIMIT):
                deny("private admission input is not an owner-private bounded file")
            data = bytearray()
            while len(data) < PRIVATE_JSON_LIMIT + 1:
                part = os.read(fd, min(65536, PRIVATE_JSON_LIMIT + 1 - len(data)))
                if not part:
                    break
                data.extend(part)
            after = os.fstat(fd)
            def identity(st):
                return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns,
                        st.st_ctime_ns)
            if len(data) != before.st_size or identity(after) != identity(before):
                deny("private admission input changed during descriptor read")
            if (absolute.resolve(strict=True) != intended or
                    intended.is_relative_to(root)):
                deny("private admission pathname changed during read")
            final_fd = _open_no_symlink_ancestors(absolute)
            try:
                last = os.fstat(final_fd)
                if (last.st_dev, last.st_ino) != (before.st_dev, before.st_ino):
                    deny("private admission inode changed during read")
            finally:
                os.close(final_fd)
        finally:
            os.close(fd)
    except OSError:
        deny("private admission file unavailable or unsafe")
    try:
        result = json.loads(data.decode("utf-8"),
                            object_pairs_hook=_unique_private_fields,
                            parse_constant=_invalid_private_constant,
                            parse_float=_finite_private_float)
    except (ValueError, UnicodeError, RecursionError, OverflowError):
        deny("private admission JSON malformed or ambiguous")
    if not isinstance(result, dict):
        deny("private admission JSON must be an object")
    return result


def _read_pinned_public_manifest():
    """Hash and parse precisely the SAME anchored public manifest bytes.

    This public file has no RACE passage, human response, or private key.
    Only its exact preregistered bytes may bind the private roster packet.
    """
    from exp053_reviewer_roster import _open_no_symlink_ancestors
    if not all(getattr(os, flag, 0) for flag in
               ("O_PATH", "O_DIRECTORY", "O_NOFOLLOW", "O_NONBLOCK")):
        deny("immutable public manifest cannot be opened safely")
    if (not isinstance(MANIFEST_SHA, str) or len(MANIFEST_SHA) != 64 or
            any(c not in "0123456789abcdef" for c in MANIFEST_SHA)):
        deny("immutable public manifest has invalid pinned digest")
    absolute = Path(os.path.abspath(os.fspath(MANIFEST)))
    try:
        intended = absolute.resolve(strict=True)
        fd = _open_no_symlink_ancestors(absolute)
        try:
            before = os.fstat(fd)
            if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or
                    not 1 <= before.st_size <= PUBLIC_MANIFEST_LIMIT):
                deny("immutable public manifest is not a bounded single-link file")
            payload = bytearray()
            while len(payload) < PUBLIC_MANIFEST_LIMIT + 1:
                part = os.read(fd, min(65536, PUBLIC_MANIFEST_LIMIT + 1 - len(payload)))
                if not part:
                    break
                payload.extend(part)
            after = os.fstat(fd)
            def identity(st):
                return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns,
                        st.st_ctime_ns)
            if len(payload) != before.st_size or identity(before) != identity(after):
                deny("immutable public manifest changed while reading")
            if hashlib.sha256(payload).hexdigest() != MANIFEST_SHA:
                deny("immutable public manifest is absent or modified")
            if absolute.resolve(strict=True) != intended:
                deny("immutable public manifest pathname changed while reading")
            final_fd = _open_no_symlink_ancestors(absolute)
            try:
                last = os.fstat(final_fd)
                if (last.st_dev, last.st_ino) != (before.st_dev, before.st_ino):
                    deny("immutable public manifest inode changed while reading")
            finally:
                os.close(final_fd)
        finally:
            os.close(fd)
    except OSError:
        deny("immutable public manifest unavailable or unsafe")
    try:
        manifest = json.loads(payload.decode("utf-8"),
                              object_pairs_hook=_unique_private_fields,
                              parse_constant=_invalid_private_constant,
                              parse_float=_finite_private_float)
    except (ValueError, UnicodeError, RecursionError, OverflowError):
        deny("immutable public manifest cannot be parsed safely")
    if not isinstance(manifest, dict):
        deny("immutable public manifest must be a JSON object")
    return manifest


SOURCE_MAX_BYTES = 64_000_000


def corpus():
    """Bind the pinned source SHA and Parquet decode to ONE file descriptor.

    A Hugging Face snapshot may be a symlink to a content-addressed blob.
    Resolve it, securely open that one target, hash and decode from its fd,
    and refuse if the lexical pointer or underlying file changes.
    """
    from exp053_reviewer_roster import _open_no_symlink_ancestors
    if (not isinstance(SOURCE_SHA, str) or len(SOURCE_SHA) != 64 or
            any(c not in "0123456789abcdef" for c in SOURCE_SHA)):
        deny("source authority digest invalid")
    if filehash(PROTOCOL) != PROTOCOL_SHA:
        deny("preregistered protocol drift")
    if not all(getattr(os, key, 0) for key in
               ("O_PATH", "O_DIRECTORY", "O_NOFOLLOW", "O_NONBLOCK")):
        deny("secure source file traversal unavailable")
    try:
        import pyarrow.parquet as pq
    except ImportError:
        deny("source parquet backend unavailable")
    try:
        resolved = SOURCE.resolve(strict=True)
        fd = _open_no_symlink_ancestors(resolved)
        with os.fdopen(fd, "rb", closefd=True) as source:
            before = os.fstat(source.fileno())
            if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or
                    not 1 <= before.st_size <= SOURCE_MAX_BYTES):
                deny("source is not a bounded single-link regular file")
            hasher = hashlib.sha256()
            counted = 0
            while True:
                data = source.read(min(65536, SOURCE_MAX_BYTES + 1 - counted))
                if not data:
                    break
                counted += len(data)
                hasher.update(data)
                if counted > SOURCE_MAX_BYTES:
                    deny("source exceeds bounded size")
            after_hash = os.fstat(source.fileno())
            def identity(st):
                return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns,
                        st.st_ctime_ns)
            if (counted != before.st_size or identity(before) != identity(after_hash) or
                    hasher.hexdigest() != SOURCE_SHA):
                deny("source bytes do not match the pinned authority")
            source.seek(0)
            # Native PyArrow may keep threaded reader handles live after the
            # Python file closes, causing a process-shutdown abort. A bounded,
            # serial read owns this fd for the entire decoded table lifetime.
            items = pq.read_table(source, use_threads=False).to_pylist()
            after_decode = os.fstat(source.fileno())
            if identity(before) != identity(after_decode):
                deny("source changed during Parquet decoding")
            if SOURCE.resolve(strict=True) != resolved:
                deny("source snapshot symlink changed during decoding")
            final_fd = _open_no_symlink_ancestors(resolved)
            try:
                last = os.fstat(final_fd)
                if (last.st_dev, last.st_ino) != (before.st_dev, before.st_ino):
                    deny("source resolved inode changed during decoding")
            finally:
                os.close(final_fd)
    except OSError:
        deny("source snapshot cannot be read safely")
    if len(items) < 2500:
        deny("insufficient test corpus")
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
    required={"schema","reviewer_id","role","packet_sha256","attestation",
              "decisions","signature"}
    if not isinstance(review,dict) or set(review)!=required:
        deny("review fields missing or unexpected")
    if not isinstance(trust,dict) or len(trust) not in (2,3):
        deny("private reviewer trust store must contain two signers and optional adjudicator")
    if review["schema"]!="exp053-review-v1":deny("unexpected review schema")
    rid=review["reviewer_id"]
    if not isinstance(rid,str) or rid not in trust:deny("untrusted reviewer key")
    if review["role"] not in ("independent","adjudicator"):
        deny("invalid signed review role")
    if review.get("attestation")!="I independently reviewed these source passages and responses; no model predictions or other reviewer labels were provided to me.":
        deny("missing independent human review attestation")
    # Direct callers must not bypass owner-roster uniqueness by using
    # alternative base64 strings for the SAME bytes. Refuse any malformed or
    # weak second key before opening an optional signature verification backend.
    from exp053_reviewer_roster import _prime_order_ed25519_public_key
    decoded={}
    used_public=set()
    for identity, encoded in trust.items():
        if not isinstance(identity,str) or not identity or not isinstance(encoded,str):
            deny("trust registry identities and keys must be nonempty strings")
        try:
            raw=base64.b64decode(encoded,validate=True)
        except (ValueError,TypeError) as exc:
            deny("invalid reviewer key encoding: "+type(exc).__name__)
        if (len(raw)!=32 or base64.b64encode(raw).decode()!=encoded or
                not _prime_order_ed25519_public_key(raw)):
            deny("trust registry contains an invalid reviewer public key")
        if raw in used_public:
            deny("trust registry contains a reused reviewer public key")
        used_public.add(raw)
        decoded[identity]=raw
    sig=review["signature"]
    if (not isinstance(sig,str) or len(sig)!=128 or
            any(c not in "0123456789abcdef" for c in sig)):
        deny("signed reviewer signature must be canonical lowercase hex")
    try:
        raw_public=decoded[rid]
        # Hosted research CI intentionally does not install optional signing
        # dependencies. Invalid trust must HOLD before importing the backend.
        try:
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        except ImportError:
            deny("optional Ed25519 signature verification backend unavailable")
        pub=Ed25519PublicKey.from_public_bytes(raw_public)
        signature=bytes.fromhex(sig)
        unsigned={k:v for k,v in review.items() if k!="signature"}
        pub.verify(signature,canonical(unsigned))
    except Exception as exc:
        deny("signature validation failed: "+type(exc).__name__)
    return rid
def require_owner_assigned_signer(review, trust, slots, slot):
    """Verify a signature AND its owner-approved blinded reviewer slot."""
    if slot not in slots:
        deny("signer slot was not independently owner-enrolled")
    rid=verify_signed(review,trust)
    if rid!=slots[slot]["reviewer_id"] or review.get("role")!=slots[slot]["role"]:
        deny("signed reviewer identity or role does not match the enrolled blind slot")
    return rid

def check_decision(dec,case):
    if not isinstance(dec,dict):deny("malformed decision entry")
    status=dec.get("decision")
    if status=="reject":
        reason=dec.get("reject_reason")
        if not isinstance(reason,str) or len(reason.strip())<12:
            deny("unreasoned reject")
        return status
    if status!="accept":deny("unreviewed or invalid decision")
    a=dec.get("supported_option");f=dec.get("foil_option")
    if not isinstance(a,str) or not isinstance(f,str) or a not in LETTERS or f not in LETTERS or a==f:
        deny("invalid chosen supported/foil options")
    quote=dec.get("support_quote")
    if not isinstance(quote,str) or len(quote.strip())<20 or quote.strip() not in case["article"]:
        deny("source quote absent or too short")
    reason=dec.get("foil_reason")
    if not isinstance(reason,str) or len(reason.strip())<15:
        deny("foil reason missing")
    return status
def decisions_for(review,packet,required_ids=None):
    if not isinstance(review,dict):deny("malformed review payload")
    if review.get("packet_sha256")!=digest(packet):deny("packet mismatch")
    if review.get("role") not in ("independent","adjudicator"):deny("invalid signer role")
    lookup={c["case_id"]:c for c in packet["cases"]}
    ds=review.get("decisions",[])
    expected=set(lookup) if required_ids is None else set(required_ids)
    if not isinstance(ds,list) or len(ds)!=len(expected):deny("incomplete review form")
    out={}
    for d in ds:
        if not isinstance(d,dict):deny("malformed reviewer decision payload")
        cid=d.get("case_id")
        if cid in out or cid not in lookup:deny("duplicate/foreign case")
        check_decision(d,lookup[cid]);out[cid]=d
    if set(out)!=expected:deny("unexpected decision scope")
    return out
def verified_admission(private_dir,keys_path,adjudicator_file=None):
    # There is no scientific authorization without an independently approved
    # owner roster. Refuse BEFORE hashing licensed source bytes or opening
    # private reviewer/answer-key files when the pin is still unset.
    pin = APPROVED_REVIEWER_ROSTER_SHA256
    if (not isinstance(pin, str) or len(pin) != 64 or
            any(c not in "0123456789abcdef" for c in pin)):
        deny("owner has not approved any independent human reviewer roster")
    # The source-free public manifest is itself pinned by the preregistered
    # SHA. Its packet digest is the trust anchor for checking the separately
    # owner-pinned private A/B roster BEFORE reading licensed source bytes.
    manifest=_read_pinned_public_manifest()
    packet_pin=manifest.get("packet_sha256") if isinstance(manifest,dict) else None
    if (not isinstance(packet_pin,str) or len(packet_pin)!=64 or
            any(c not in "0123456789abcdef" for c in packet_pin)):
        deny("immutable public manifest has invalid packet identity")
    from exp053_reviewer_roster import load_roster
    roster=load_roster(keys_path,packet_pin,pin,
                       Path(__file__).resolve().parents[1])
    # A syntactically valid roster pin alone is not proof of human enrollment.
    # Only exact, owner-pinned bytes with matching packet identity allow the
    # existing SOURCE/protocol/private packet/sealed-answer checks to start.
    if not SOURCE.is_file() or filehash(SOURCE)!=SOURCE_SHA:deny("source authority changed")
    if not PROTOCOL.is_file() or filehash(PROTOCOL)!=PROTOCOL_SHA:deny("protocol drift")
    packet=_read_private_admission_json(private_dir/"packet.json")
    seal=_read_private_admission_json(private_dir/"sealed_exam_key.json")
    if packet.get("source_sha256")!=SOURCE_SHA or packet.get("protocol_sha256")!=PROTOCOL_SHA:
        deny("packet custody mismatch")
    if digest(packet)!=manifest["packet_sha256"]:deny("packet differs from manifest")
    if (seal["packet_sha256"]!=digest(packet) or
        digest(seal["answer_key"])!=manifest["sealed_answer_key_sha256"]):
        deny("answer key differs from committed manifest")
    trust=roster["keys"]
    observations=[];signer_ids=[]
    for slot in ("A","B"):
        review=_read_private_admission_json(private_dir/("reviewer_"+slot+".json"))
        rid=require_owner_assigned_signer(review,trust,roster["slots"],slot)
        signer_ids.append(rid)
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
        if "ADJ" not in roster["slots"]:
            deny("third independent adjudicator has not been owner-enrolled")
        third=_read_private_admission_json(adjudicator_file)
        third_id=require_owner_assigned_signer(third,trust,roster["slots"],"ADJ")
        if third_id in signer_ids:
            deny("adjudicator must be independent of both reviewers")
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
    return {"schema":"exp053-admitted-v1","packet_sha256":digest(packet),
            "reviewers":signer_ids,"accepted":accepted,"rejected":rejected,
            "owner_roster_sha256":roster["roster_sha256"],
            "owner_custody_reference_sha256":roster["owner_custody_reference_sha256"],
            "human_approval":True,"simulation":False}

def gate(private_dir,keys_path,output,adjudicator_file=None):
    if output.exists():deny("gate output exists")
    receipt=verified_admission(private_dir,keys_path,adjudicator_file)
    save_private(output,receipt)
    return len(receipt["accepted"])

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
