#!/usr/bin/env python3
"""EXP053 verified model-free scoring replay; no model imports or inference."""
import argparse
import hashlib
import json
import os
import stat
from pathlib import Path
from exp053_race_review import (
    MANIFEST as MANIFEST, MANIFEST_SHA as MANIFEST_SHA,
    digest, filehash, verified_admission, deny,
    _read_pinned_public_manifest, _read_private_admission_json,
    _unique_private_fields, _invalid_private_constant, _finite_private_float,
)
from exp053_tokenizer_preflight import PRIVATE,PATHS,ROOT,WITHHELD,options_for,prompt
from exp053_metrics import analyze, FIELDS as SCORE_FIELDS, rotate as rotate_label
from exp053_execution_attestation import (
    verify_attestation, read_attested_scores, private_regular_file,
)
from exp053_attempt_ledger import verify_attempt_ledger
from exp053_operator_run import consume_unique_run

PREFLIGHT=ROOT/"EXP053-TOKENIZER-PREFLIGHT-2026-10-16.json"
PREFLIGHT_SHA="8e35a44e1ebb725c0b52469baa4ab5d41472e38fe55346352b96dd26697726be"
PROMPT_CODE=ROOT/"exp053_tokenizer_preflight.py"
PROMPT_CODE_SHA="cf138946680db1c1259edff4ac895c4492992b69edb275d3c3c91e7e39f44664"
WEIGHT_REPORT=ROOT/"EXP053-WEIGHT-IDENTITY-2026-10-16.json"
# The operator must independently review and pin a real file-byte receipt.
# It remains intentionally unset: no score report can be admitted yet.
WEIGHT_REPORT_SHA=""
WEIGHT_SPEC=ROOT/"EXP053-WEIGHT-ID-PREFLIGHT-SPEC-2026-10-16.md"
WEIGHT_SPEC_SHA="b89f573acc2567b114e65746b3026e36636aee9add6c6568c0fac6319f620a81"

PUBLIC_RECEIPT_LIMIT = 1_000_000


def _read_pinned_source_free_receipt(path, expected_sha):
    """Verify and parse exactly the SAME bounded public metadata FD bytes."""
    from exp053_reviewer_roster import _open_no_symlink_ancestors
    if not hex_digest(expected_sha):
        deny("source-free receipt pin missing or invalid")
    if not all(getattr(os, flag, 0) for flag in
               ("O_PATH", "O_DIRECTORY", "O_NOFOLLOW", "O_NONBLOCK")):
        deny("safe source-free receipt descriptor access unavailable")
    absolute = Path(os.path.abspath(os.fspath(path)))
    try:
        intended = absolute.resolve(strict=True)
        fd = _open_no_symlink_ancestors(absolute)
        try:
            before = os.fstat(fd)
            if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or
                    not 1 <= before.st_size <= PUBLIC_RECEIPT_LIMIT):
                deny("source-free receipt is not a bounded regular file")
            data = bytearray()
            while len(data) < PUBLIC_RECEIPT_LIMIT + 1:
                chunk = os.read(fd, min(65536, PUBLIC_RECEIPT_LIMIT + 1 - len(data)))
                if not chunk:
                    break
                data.extend(chunk)
            after = os.fstat(fd)
            def identity(st):
                return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns,
                        st.st_ctime_ns)
            if len(data) != before.st_size or identity(before) != identity(after):
                deny("source-free receipt changed during read")
            if hashlib.sha256(data).hexdigest() != expected_sha:
                deny("pinned source-free receipt bytes do not match")
            if absolute.resolve(strict=True) != intended:
                deny("source-free receipt pathname changed during read")
            final_fd = _open_no_symlink_ancestors(absolute)
            try:
                newest = os.fstat(final_fd)
                if (newest.st_dev, newest.st_ino) != (before.st_dev, before.st_ino):
                    deny("source-free receipt inode changed during read")
            finally:
                os.close(final_fd)
        finally:
            os.close(fd)
    except OSError:
        deny("source-free receipt unavailable or unsafe")
    try:
        value = json.loads(data.decode("utf-8"),
                           object_pairs_hook=_unique_private_fields,
                           parse_constant=_invalid_private_constant,
                           parse_float=_finite_private_float)
    except (ValueError, UnicodeError, RecursionError, OverflowError):
        deny("source-free receipt JSON malformed or ambiguous")
    if not isinstance(value, dict):
        deny("source-free receipt JSON must be an object")
    return value


def sha(s):return hashlib.sha256(s.encode("utf8")).hexdigest()
def hex_digest(s):return isinstance(s,str) and len(s)==64 and all(c in "0123456789abcdef" for c in s)

def admitted_weight_inventory(scan):
    """Validate operator-pinned checkpoint-file receipts; never read neural weights."""
    from exp053_weight_identity import sha as identity_sha
    if not WEIGHT_SPEC.is_file() or filehash(WEIGHT_SPEC)!=WEIGHT_SPEC_SHA:
        deny("frozen checkpoint identity protocol changed")
    if not WEIGHT_REPORT_SHA:
        deny("HOLD no independently accepted checkpoint file-identity receipt")
    report=_read_pinned_source_free_receipt(WEIGHT_REPORT, WEIGHT_REPORT_SHA)
    if (report.get("schema")!="exp053-source-only-weight-identity-v1" or
            report.get("spec_sha256")!=WEIGHT_SPEC_SHA or
            report.get("tokenizer_receipt_sha256")!=PREFLIGHT_SHA or
            report.get("neural_forward_passes")!=0 or
            report.get("human_reviewed_articles")!=0):
        deny("unexpected checkpoint inventory schema or source authority")
    models=report.get("models")
    if not isinstance(models,dict) or set(models)!=set(scan["models"]):
        deny("incomplete checkpoint inventory")
    if set(report)!={"schema","spec_sha256","tokenizer_receipt_sha256",
                     "models","neural_forward_passes","human_reviewed_articles",
                     "checkpoint_receipts_sha256","claim_limit"}:
        deny("unexpected weight-report fields, potentially private material")
    if report["claim_limit"]!="File-byte identity only; no proof of runtime execution or grounding.":
        deny("incorrect weight receipt scientific claim scope")
    if identity_sha(models)!=report.get("checkpoint_receipts_sha256"):
        deny("checkpoint inventory aggregate mismatch")
    identities={}
    for name,rec in models.items():
        known=scan["models"][name]
        if (rec.get("model")!=name or
                rec.get("config_sha256")!=known["config_sha256"] or
                rec.get("tokenizer_sha256")!=known["tokenizer_json_sha256"] or
                rec.get("checkpoint_revision_or_directory")!=known["checkpoint_directory"]):
            deny("checkpoint inventory not tied to frozen tokenizer/config")
        expected_fields={"model","checkpoint_revision_or_directory","config_sha256",
                         "tokenizer_sha256","weight_index_sha256","shards","shard_count",
                         "total_weight_bytes","checkpoint_receipt_sha256"}
        if not isinstance(rec,dict) or set(rec)!=expected_fields:
            deny("unexpected checkpoint record fields")
        from exp053_weight_identity import TWO_SHARD_MODELS
        index_hash=rec["weight_index_sha256"]
        if name in TWO_SHARD_MODELS:
            if not hex_digest(index_hash):deny("invalid checkpoint weight-map identity")
        elif index_hash is not None:
            deny("unexpected index file for single-file checkpoint")
        shards=rec["shards"]
        if (not isinstance(shards,list) or
                (name in TWO_SHARD_MODELS and len(shards)<2) or
                (name not in TWO_SHARD_MODELS and len(shards)!=1)):
            deny("missing or unexpected checkpoint shards")
        filenames=[]
        size=0
        for entry in shards:
            if not isinstance(entry,dict) or set(entry)!={"filename","bytes","sha256"}:
                deny("unexpected checkpoint shard fields")
            filename=entry["filename"]
            if (not isinstance(filename,str) or not filename.endswith(".safetensors")
                    or "/" in filename or "\\" in filename or
                    filename.startswith(".") or ".." in filename):
                deny("invalid checkpoint shard filename")
            if type(entry["bytes"])!=int or entry["bytes"]<4096 or not hex_digest(entry["sha256"]):
                deny("invalid checkpoint shard identity")
            filenames.append(filename)
            size+=entry["bytes"]
        if (filenames!=sorted(set(filenames)) or
                type(rec["shard_count"])!=int or rec["shard_count"]!=len(shards) or
                type(rec["total_weight_bytes"])!=int or rec["total_weight_bytes"]!=size):
            deny("checkpoint shard inventory count, order or size mismatch")
        if not hex_digest(rec["checkpoint_receipt_sha256"]):
            deny("invalid checkpoint receipt checksum")
        calculated=identity_sha({k:v for k,v in rec.items() if k!="checkpoint_receipt_sha256"})
        if calculated!=rec.get("checkpoint_receipt_sha256"):
            deny("individual checkpoint digest mismatch")
        identities[name]=calculated
    return identities

def require_sources(private_dir=None):
    scan=_read_pinned_source_free_receipt(PREFLIGHT, PREFLIGHT_SHA)
    if not PROMPT_CODE.is_file() or filehash(PROMPT_CODE)!=PROMPT_CODE_SHA:
        deny("prompt builder source drift")
    manifest=_read_pinned_public_manifest()
    if set(scan["models"])!=set(PATHS) or scan["model_forward_passes"]!=0:
        deny("tokenizer feasibility report invalid")
    source_packet=(PRIVATE if private_dir is None else Path(private_dir)/"packet.json")
    packet=_read_private_admission_json(source_packet)
    if digest(packet)!=manifest["packet_sha256"]:
        deny("private packet does not match the public hash")
    return packet,manifest,scan

def check_private_archive(rows,approved,packet,scan,weight_ids):
    """Reconstruct every source/label receipt; standalone validation is not approval."""
    if not isinstance(rows,list) or not isinstance(approved,dict):
        deny("expected private score rows and signed-approved case map")
    if not isinstance(weight_ids,dict) or set(weight_ids)!=set(scan["models"]):
        deny("independently pinned checkpoint identities must cover every model")
    cases={c["case_id"]:c for c in packet["cases"]}
    if len(cases)!=len(packet["cases"]):
        deny("duplicate case identity in the private cohort")
    ids=sorted(approved)
    if len(ids)<40 or not set(ids)<=set(cases):
        deny("fewer than 40 signed approved in-cohort cases")
    next_case={c:ids[(i+1)%len(ids)] for i,c in enumerate(ids)}
    for record in rows:
        if not isinstance(record,dict) or set(record)!=SCORE_FIELDS:
            deny("scorer receipt missing fields or leaking private data")
        cid=record["case_id"];rot=record["rotation"];arm=record["arm"]
        if (not isinstance(cid,str) or cid not in approved or
                type(rot)!=int or rot not in (0,1)):
            deny("unapproved or invalid scored case")
        if not isinstance(arm,str) or arm not in ("supported","withheld","decoy"):
            deny("invalid source arm")
        model=record["model"]
        if not isinstance(model,str) or model not in scan["models"]:
            deny("unknown score model")
        case=cases[cid]
        if record["article_sha256"]!=sha(case["article"]):
            deny("score article digest differs from source article")
        opts=options_for(case,rot)
        if record["option_order_sha256"]!=digest(opts):
            deny("displayed options do not match the frozen packet and rotation")
        passage=(case["article"] if arm=="supported" else
                 WITHHELD if arm=="withheld" else cases[next_case[cid]]["article"])
        expected_context=("WITHHELD" if arm=="withheld" else sha(passage))
        if record["context_article_sha256"]!=expected_context:
            deny("source-arm context digest differs from exact source passage")
        if record["prompt_sha256"]!=sha(prompt(case["question"],opts,passage)):
            deny("prompt hash does not match source, question, options and decoy")
        human=approved[cid]
        if not isinstance(human,dict) or set(human)!={"supported","foil"}:
            deny("signed case verdict has unexpected fields")
        good,foil=human["supported"],human["foil"]
        if (not isinstance(good,str) or good not in "ABCD" or
                not isinstance(foil,str) or foil not in "ABCD" or good==foil):
            deny("invalid signed gold and foil option")
        if rot==1:
            good,foil=rotate_label(good),rotate_label(foil)
        if (record["gold_label"],record["foil_label"])!=(good,foil):
            deny("score labels disagree with signed approved foil and rotation")
        if record["tokenizer_sha256"]!=scan["models"][model]["tokenizer_json_sha256"]:
            deny("tokenizer does not match the frozen preflight checkpoint")
        if record["checkpoint_sha256"]!=weight_ids[model]:
            deny("score weights do not match the independently pinned checkpoint")
        if (type(record["prompt_token_count"])!=int or
                not 1<=record["prompt_token_count"]<=4096 or
                type(record["gold_token_count"])!=int or record["gold_token_count"]!=1 or
                type(record["foil_token_count"])!=int or record["foil_token_count"]!=1):
            deny("invalid score tokenization counts")
        if record["mode"]!="raw":
            deny("only separate base-mode score receipts are permitted")
    return len(rows)

def replay(private_dir,trust_roots,score_archive,output,adjudicator=None,
           execution_attestation=None,attempt_log=None,operator_authorization=None):
    if not output.parent.is_dir() or output.parent.resolve()!=ROOT.resolve() or output.exists():
        deny("public aggregate output must be a new file inside research")
    if not score_archive.resolve().is_relative_to(private_dir.resolve()):
        deny("individual score archive must be private")
    admission=verified_admission(private_dir,trust_roots,adjudicator)
    packet,manifest,scan=require_sources(private_dir)
    if admission["packet_sha256"]!=digest(packet):deny("admission packet mismatch")
    checkpoint_ids=admitted_weight_inventory(scan)
    if execution_attestation is None or attempt_log is None:
        deny("operator-attested scorer execution and private retry log required")
    provenance=verify_attestation(
        execution_attestation,score_archive,attempt_log,private_dir,
        packet_sha256=digest(packet),admission_sha256=digest(admission),
        weight_inventory_sha256=WEIGHT_REPORT_SHA,
        tokenizer_preflight_sha256=PREFLIGHT_SHA,
        prompt_builder_sha256=PROMPT_CODE_SHA,checkpoint_receipts=checkpoint_ids,
        approved_count=len(admission["accepted"]))
    # Parse the exact bytes hashed through one private file descriptor, not
    # a second path read vulnerable to swap-then-restore between hash checks.
    rows=read_attested_scores(
        score_archive,private_dir,provenance["score_archive_sha256"])
    check_private_archive(rows,admission["accepted"],packet,scan,checkpoint_ids)
    attempt_evidence=verify_attempt_ledger(
        attempt_log,private_dir,rows,
        execution_id_sha256=provenance["execution_id_sha256"],
        packet_sha256=digest(packet),
        attempt_log_sha256=provenance["attempt_log_sha256"])
    summary=analyze(rows,admission["accepted"],manifest)
    summary["score_archive_sha256"]=provenance["score_archive_sha256"]
    summary["signed_admission_sha256"]=digest(admission)
    summary["tokenizer_preflight_sha256"]=PREFLIGHT_SHA
    summary["checkpoint_receipt_sha256"]=WEIGHT_REPORT_SHA
    summary["execution_attestation_sha256"]=provenance["execution_attestation_sha256"]
    summary["attempt_log_sha256"]=provenance["attempt_log_sha256"]
    summary["attempt_cells"]=attempt_evidence["attempt_cells"]
    summary["total_logged_attempts"]=attempt_evidence["total_logged_attempts"]
    summary["logged_failed_attempts"]=attempt_evidence["logged_failed_attempts"]
    summary["runner_source_sha256"]=provenance["runner_source_sha256"]
    summary["privacy"]="aggregates only; individual scores and RACE text stay private"
    # Refuse score/log mutation during downstream statistical calculation;
    # only this source-free aggregate may be persisted to public research.
    score_end,_=private_regular_file(score_archive,private_dir)
    attempts_end,_=private_regular_file(attempt_log,private_dir)
    if (score_end!=provenance["score_archive_sha256"] or
            attempts_end!=provenance["attempt_log_sha256"]):
        deny("private scorer evidence changed during aggregate calculation")
    if operator_authorization is None:
        deny("independent operator-issued single-use run authorization required")
    # Claim only after ALL source, score, retry and statistical checks pass.
    # Claim remains consumed if later output persistence fails (fail closed).
    operator_claim=consume_unique_run(
        operator_authorization,private_dir,provenance=provenance,
        packet_sha256=digest(packet))
    summary.update(operator_claim)
    with output.open("x") as f:f.write(json.dumps(summary,sort_keys=True,indent=2)+"\n")
    return summary

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--private-dir",type=Path,required=True)
    p.add_argument("--trust-roots",type=Path,required=True)
    p.add_argument("--private-score-archive",type=Path,required=True)
    p.add_argument("--private-execution-attestation",type=Path,required=True)
    p.add_argument("--private-attempt-log",type=Path,required=True)
    p.add_argument("--private-operator-authorization",type=Path,required=True)
    p.add_argument("--adjudicator-file",type=Path)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(replay(a.private_dir,a.trust_roots,a.private_score_archive,
                            a.out,a.adjudicator_file,a.private_execution_attestation,
                            a.private_attempt_log,a.private_operator_authorization),sort_keys=True))
if __name__=="__main__":main()
