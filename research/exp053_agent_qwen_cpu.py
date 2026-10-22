#!/usr/bin/env python3
"""Offline, CPU-only EXP-053 machine-proxy feasibility, resumable and fail-closed.

This is NOT the independently human-adjudicated experiment. It produces one
immutable owner-private receipt per original RACE case and nothing in Git.
"""
import argparse
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import stat
import sys
import time

from exp053_agent_only import machine_receipt
from exp053_tokenizer_preflight import LETTERS, PATHS, PRIVATE, prompts, validate_custody

NAME = "Qwen3.5-4B-Base"
SCHEMA = "exp053-qwen-machine-proxy-case-v2"
OUT_ROOT = Path("/media/scott/SSD_4TB/exp053-private-20261016")
RECEIPTS = OUT_ROOT / "qwen-machine-proxy-20261022-v2"
FROZEN = Path(__file__).resolve().parent / "EXP053-WEIGHT-IDENTITY-2026-10-16.json"
ARM_ORDER = ("supported", "withheld", "decoy")


def fail(reason):
    raise RuntimeError("HOLD " + reason)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def digest_file(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def private_dir(path):
    if path.is_symlink() or not path.is_dir():
        fail("private root missing or symlinked")
    s = path.stat()
    if s.st_uid != os.getuid() or stat.S_IMODE(s.st_mode) & 0o077:
        fail("private path not owner-only")
    return path


def read_receipt(path):
    if path.is_symlink() or not path.is_file():
        fail("existing case receipt is not regular")
    info = path.stat()
    if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o600:
        fail("existing receipt permissions differ")
    if info.st_size > 32768:
        fail("oversized existing receipt")
    return json.loads(path.read_bytes())


def create_immutable(directory, filename, content):
    """Write-then-publish with link(O_EXCL semantics): never leave a partial final."""
    data = (json.dumps(content, sort_keys=True, separators=(",", ":")) + "\n").encode()
    if len(data) > 32768:
        fail("receipt exceeded size bound")
    temp = ".pending-" + str(os.getpid()) + "-" + str(time.time_ns())
    dfd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                     0o600, dir_fd=dfd)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            # Directory-relative hard-link publishes only if filename is still absent.
            os.link(temp, filename, src_dir_fd=dfd, dst_dir_fd=dfd, follow_symlinks=False)
            os.fsync(dfd)
        finally:
            os.unlink(temp, dir_fd=dfd)
            os.fsync(dfd)
    finally:
        os.close(dfd)
    return sha(data)


def save_case(directory, index, content):
    return create_immutable(directory, "case-%03d.json" % index, content)


def journal_failure(directory, index, failure):
    # No prompts, answer text, paths, scores, exception messages or credentials.
    record = {"schema":"exp053-qwen-machine-failure-v1", "case_index":index,
              "failure_type":type(failure).__name__,
              "machine_only":True,"human_adjudicated":False,
              "status":"INCOMPLETE_DO_NOT_USE_AS_SCORE"}
    create_immutable(directory,"failure-%03d-%d.json" % (index,time.time_ns()),record)


def verify_case(value, index, cid, checkpoint, packet, runner_sha):
    if not isinstance(value, dict) or value.get("schema") != SCHEMA:
        fail("existing receipt schema")
    if value.get("cohort_index") != index or value.get("case_id") != cid:
        fail("existing receipt case mismatch")
    if value.get("checkpoint_receipt_sha256") != checkpoint:
        fail("existing receipt model identity mismatch")
    if value.get("private_packet_sha256") != packet:
        fail("existing receipt source packet mismatch")
    if value.get("runner_sha256") != runner_sha:
        fail("existing receipt uses a different runner source revision")
    if value.get("machine_only") is not True or value.get("human_adjudicated") is not False:
        fail("existing receipt claim boundary")
    rows = value.get("conditions", [])
    if not isinstance(rows, list) or len(rows) != 6:
        fail("incomplete old case receipt")
    pairs = [(r.get("arm"), r.get("rotation")) for r in rows]
    if pairs != [(arm, rot) for rot in (0, 1) for arm in ARM_ORDER]:
        fail("existing receipt condition order")
    if len({r.get("prompt_sha256") for r in rows}) != 6:
        fail("existing receipt prompt collision")
    if any(r.get("gold_label") not in LETTERS or r.get("foil_label") not in LETTERS
           or r["gold_label"] == r["foil_label"] for r in rows):
        fail("existing receipt invalid answer labels")
    if any(type(r.get(k)) != float or not math.isfinite(r[k])
           for r in rows for k in ("gold_logp", "foil_logp")):
        fail("existing nonfinite logprob")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--count", type=int, default=4)
    p.add_argument("--execute-source-backed", action="store_true")
    a = p.parse_args()
    if not (0 <= a.start < 64 and 1 <= a.count <= 64 and a.start + a.count <= 64):
        fail("selection must be 1..64 source-disjoint pinned cases")
    if os.geteuid() == 0:
        fail("root execution forbidden")
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        fail("offline flags missing")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        fail("GPU must be hidden for isolated CPU evaluation")
    private_dir(OUT_ROOT)
    if PRIVATE.parent.resolve() != OUT_ROOT.resolve():
        fail("pinned packet not in fixed owner-private root")
    lookup, manifest = validate_custody()
    proxy = machine_receipt(PRIVATE.parent)
    ids = sorted(lookup)
    if len(ids) != 64 or set(proxy["accepted"]) != set(ids):
        fail("frozen cohort and proxy do not match")
    frozen = json.loads(FROZEN.read_bytes())
    checkpoint = frozen["models"][NAME]
    model_dir = PATHS[NAME]
    if sha((model_dir / "config.json").read_bytes()) != checkpoint["config_sha256"]:
        fail("config digest mismatch")
    if sha((model_dir / "tokenizer.json").read_bytes()) != checkpoint["tokenizer_sha256"]:
        fail("tokenizer digest mismatch")
    shards = checkpoint["shards"]
    if len(shards) != 2:
        fail("expected two pinned BF16 shards")
    for shard in shards:
        fn = model_dir / shard["filename"]
        if not fn.is_file() or fn.stat().st_size != shard["bytes"] or digest_file(fn) != shard["sha256"]:
            fail("exact safetensors content fingerprint mismatch")
    selection = ids[a.start:a.start+a.count]
    # Preconstruct all private prompts from frozen code, before any inference.
    prompts_by_case = {cid: [] for cid in selection}
    for cid, rot, arm, prompt in prompts(lookup):
        if cid in prompts_by_case:
            prompts_by_case[cid].append((rot, arm, prompt))
    for cid in selection:
        if len(prompts_by_case[cid]) != 6 or [(arm, rot) for rot, arm, _ in prompts_by_case[cid]] != [
             (arm, rot) for rot in (0, 1) for arm in ARM_ORDER]:
            fail("prompt grid mismatch")
    if not a.execute_source_backed:
        print("EXP053_AGENT_QWEN_PREFLIGHT_ONLY range=%d..%d cases=%d; exact_weight_shards=2; no forwards" %
              (a.start, a.start+a.count-1, len(selection)), flush=True)
        return

    RECEIPTS.mkdir(mode=0o700, exist_ok=True)
    private_dir(RECEIPTS)
    code_sha = sha(Path(__file__).read_bytes())
    pending = []
    existing = 0
    for i, cid in enumerate(selection, start=a.start):
        path = RECEIPTS / ("case-%03d.json" % i)
        if path.exists() or path.is_symlink():
            verify_case(read_receipt(path), i, cid, checkpoint["checkpoint_receipt_sha256"],
                        manifest["packet_sha256"], code_sha)
            existing += 1
        else:
            pending.append((i, cid))
    if not pending:
        print("EXP053_AGENT_QWEN_ALL_SELECTED_ALREADY_VERIFIED count=%d" % existing, flush=True)
        return

    import torch
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    if "5.12.1" != importlib.metadata.version("transformers"):
        fail("unqualified Transformers version")
    if str(torch.__version__) != "2.13.0+rocm7.1":
        fail("unqualified Torch version")
    runtime = {"python": sys.version.split()[0], "torch": str(torch.__version__),
               "transformers": importlib.metadata.version("transformers"),
               "model_impl_sha256": digest_file(Path(inspect.getfile(Qwen3_5ForConditionalGeneration)))}
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=False)
    labels = {letter: tokenizer.encode(" "+letter, add_special_tokens=False) for letter in LETTERS}
    if any(len(x)!=1 for x in labels.values()) or len({x[0] for x in labels.values()}) != 4:
        fail("answer label tokenizer must use 4 distinct 1-token continuations")
    prepared = {}
    for i, cid in pending:
        batch = []
        for rot, arm, prompt in prompts_by_case[cid]:
            token_ids = tokenizer.encode(prompt, add_special_tokens=True)
            if not (1 <= len(token_ids) <= 4096):
                fail("prompt token limit")
            if any(tokenizer.encode(prompt+" "+letter, add_special_tokens=True) != token_ids+labels[letter]
                   for letter in LETTERS):
                fail("next-token label boundary mismatch")
            batch.append((rot, arm, prompt, token_ids))
        prepared[cid] = batch

    torch.set_num_threads(2)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        str(model_dir), local_files_only=True, trust_remote_code=False, dtype=torch.bfloat16,
        device_map="cpu", attn_implementation="eager"
    )
    model.eval()
    print("EXP053_AGENT_QWEN_MODEL_READY pending_cases=%d existing_verified=%d private_only=1" %
          (len(pending), existing), flush=True)
    began = time.monotonic()
    completed = 0
    for i, cid in pending:
        label_pair = proxy["accepted"][cid]
        if label_pair["supported_exam_key"] == label_pair["other_exam_option_proxy"]:
            fail("proxy foil must differ from sealed key")
        rows = []
        try:
            for rot, arm, prompt, token_ids in prepared[cid]:
                def rotated(l): return LETTERS[(LETTERS.index(l)+2*rot)%4]
                correct = rotated(label_pair["supported_exam_key"])
                wrong = rotated(label_pair["other_exam_option_proxy"])
                with torch.inference_mode():
                    logits = model(input_ids=torch.tensor([token_ids], dtype=torch.long), use_cache=False).logits[0,-1,:].float()
                    if not bool(torch.isfinite(logits).all().item()):
                        fail("nonfinite model logits")
                    logp = torch.log_softmax(logits, dim=-1)
                    gold_logp = float(logp[labels[correct][0]].item())
                    foil_logp = float(logp[labels[wrong][0]].item())
                if not (math.isfinite(gold_logp) and math.isfinite(foil_logp) and gold_logp <= 0 and foil_logp <= 0):
                    fail("logprob bounds")
                rows.append({"arm":arm, "rotation":rot, "prompt_sha256":sha(prompt.encode()),
                             "input_tokens":len(token_ids), "gold_logp":gold_logp, "foil_logp":foil_logp,
                             "gold_label":correct, "foil_label":wrong})
        except Exception as exc:
            journal_failure(RECEIPTS, i, exc)
            raise
        payload = {"schema":SCHEMA,"cohort_index":i,"case_id":cid,
                   "private_packet_sha256":manifest["packet_sha256"],
                   "checkpoint_receipt_sha256":checkpoint["checkpoint_receipt_sha256"],
                   "tokenizer_sha256":checkpoint["tokenizer_sha256"],
                   "machine_only":True,"human_adjudicated":False,"study_complete":False,
                   "source_disjoint_heldout_claim":False,"runner_sha256":code_sha,
                   "runtime":runtime,"condition_count":6,"conditions":rows,
                   "claim_limit":"Exam-key machine proxy; no human foil adjudication or semantic validation"}
        file_sha = save_case(RECEIPTS, i, payload)
        completed += 1
        print("EXP053_AGENT_QWEN_CASE_RECEIPT_OK index=%d conditions=6 sha256=%s progress=%d/%d elapsed_s=%.1f" %
              (i,file_sha,completed,len(pending),time.monotonic()-began), flush=True)
    print("EXP053_AGENT_QWEN_SELECTION_PASS selected=%d newly_completed=%d reused=%d" %
          (len(selection),completed,existing),flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("EXP053_AGENT_QWEN_HOLD %s" % type(exc).__name__,flush=True)
        raise SystemExit(2)