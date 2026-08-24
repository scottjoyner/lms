#!/usr/bin/env python3
"""Long-context throughput sweep: measures TTFT and decode tps at increasing prompt sizes."""
import argparse, json, time, urllib.request

FILLER = ("The quarterly infrastructure review covers GPU allocation across the rendering farm, "
          "thermal envelopes for sustained inference workloads, memory bandwidth contention between "
          "concurrent model instances, scheduler fairness policies, checkpoint rotation schedules, "
          "network topology impacts on distributed training, storage tiering for cold model weights, "
          "power budget enforcement during peak hours, and capacity planning for the next fiscal year. ")

def build_prompt(target_tokens):
    words_per_rep = len(FILLER.split())
    reps = max(1, int(target_tokens / words_per_rep * 1.3))
    return FILLER * reps + "\n\nIgnore all of the above text. Reply with exactly one word: ACK"

def run(base, model, target_tokens, max_tokens=128):
    req = urllib.request.Request(
        base.rstrip("/") + "/chat/completions",
        data=json.dumps({"model": model, "stream": True,
                         "messages": [{"role": "user", "content": build_prompt(target_tokens)}],
                         "max_tokens": max_tokens, "temperature": 0}).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft = None
    chunks = 0
    usage = {}
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                d = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if d.get("usage"):
                usage = d["usage"]
            if d.get("choices") and (d["choices"][0].get("delta") or {}).get("content") is not None:
                if ttft is None:
                    ttft = time.perf_counter() - t0
                chunks += 1
    total = time.perf_counter() - t0
    ct = usage.get("completion_tokens", chunks)
    return {
        "target_prompt_tokens": target_tokens,
        "prompt_tokens": usage.get("prompt_tokens"),
        "ttft_s": round(ttft, 2) if ttft else None,
        "decode_s": round(total - (ttft or 0), 2),
        "completion_tokens": ct,
        "decode_tps": round(ct / (total - (ttft or 0)), 1) if total > (ttft or 0) else None,
        "prompt_tps_prefill": round(usage.get("prompt_tokens", 0) / ttft, 0) if ttft else None,
    }

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="http://127.0.0.1:1234/v1")
    p.add_argument("--model", required=True)
    p.add_argument("--targets", default="2000,8000,16000,32000,48000")
    a = p.parse_args()
    out = []
    for t in [int(x) for x in a.targets.split(",")]:
        r = run(a.base, a.model, t)
        r["model"] = a.model
        out.append(r)
        print(json.dumps(r), flush=True)
