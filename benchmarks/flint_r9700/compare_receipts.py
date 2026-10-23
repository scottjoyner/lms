"""Strict offline Flint quantized benchmark pairing; NO execution or promotion."""
from __future__ import annotations

import math
from run_flint import PIN, QUANTS, PROFILES


def compare(a: dict, b: dict) -> dict:
    """Never trust an exact hash *claim* as independent GPU/runtime attestation."""
    base = {"status": "HOLD", "production_admission": "NO_GO",
            "actual_model_execution_attested": False, "provider_calls": False,
            "pairs": []}
    if not isinstance(a, dict) or not isinstance(b, dict):
        return {**base, "reason": "MALFORMED_RECEIPT"}
    keys = ("host", "hf_revision", "runtime_source_sha", "profile", "runtime_argv_labels")
    if (any(a.get(k) != b.get(k) for k in keys)
            or a.get("host") != PIN["host"]
            or a.get("hf_revision") != PIN["hf_revision"]
            or a.get("runtime_source_sha") != PIN["runtime_source_sha"]
            or a.get("profile") not in PROFILES
            or not isinstance(a.get("runtime_argv_labels"), dict)
            or a["runtime_argv_labels"].get("gpu") != PIN["device"]):
        return {**base, "reason": "UNMATCHED_OR_UNPINNED_RUNTIME"}
    seen = set()
    maps = []
    for record in (a, b):
        known = next((x for x in QUANTS.values()
                      if x["quant"] == record.get("quant")), None)
        if (known is None or known["quant"] in seen
                or record.get("model_sha256_pin") != known["sha256"]
                or record.get("model_file_expected_bytes") != known["bytes"]
                or record.get("status") != "MEASURED_ISOLATED"
                or record.get("benchmark_attempted") is not True
                or record.get("model_hash_verified") is not True
                or record.get("exit_code") != 0
                or record.get("stopped_reason") is not None
                or record.get("logs_truncated") is not False
                or record.get("reasons") != []):
            return {**base, "reason": "UNVERIFIED_MEASUREMENT"}
        seen.add(known["quant"])
        bench = record.get("bench")
        if not isinstance(bench, dict) or not isinstance(bench.get("observations"), list):
            return {**base, "reason": "MALFORMED_METRICS"}
        obs = bench["observations"]
        if not obs or bench.get("valid_results") != len(obs) or len(obs) > 32:
            return {**base, "reason": "MALFORMED_METRICS"}
        samples = {}
        for row in obs:
            if not isinstance(row, dict):
                return {**base, "reason": "MALFORMED_METRICS"}
            pp, tg, speed = (row.get("n_prompt"), row.get("n_gen"),
                             row.get("avg_tokens_per_second"))
            if (type(pp) is not int or type(tg) is not int
                    or pp < 0 or tg < 0 or pp + tg == 0
                    or type(speed) not in (float, int)
                    or not math.isfinite(speed) or speed <= 0
                    or (pp, tg) in samples):
                return {**base, "reason": "MALFORMED_METRICS"}
            samples[(pp, tg)] = float(speed)
        maps.append(samples)
    if set(maps[0]) != set(maps[1]):
        return {**base, "reason": "UNPAIRED_WORKLOADS"}
    pairs = [{"n_prompt": pp, "n_gen": tg,
              "ratio_second_to_first": round(maps[1][(pp,tg)] / maps[0][(pp,tg)], 4)}
             for pp,tg in sorted(maps[0])]
    return {**base, "status": "LOCAL_RECEIPTS_STRUCTURALLY_COMPARABLE",
            "reason": "INDEPENDENT_EXECUTION_AND_QUALITY_STILL_UNVERIFIED",
            "first_quant": a["quant"], "second_quant": b["quant"], "pairs": pairs}
