"""Aggregate-only, fail-closed EXP046 retrieval vs generation scoring."""
from k2_qasper_retrieval_fixture import (
    KINDS,retrieval_audit,PAPERS)

def hit_conditioned(rows,outputs,stats):
    if len(rows)!=60 or len(outputs)!=60 or len(stats)!=PAPERS:
        raise RuntimeError("HOLD incomplete query-retrieval pairing")
    out={}
    for kind,key in (("bm25_answer_top1","answer_gold_top1"),
                     ("bm25_answer_top3","answer_gold_top3")):
        for present in (False,True):
            chosen=[val for row,val in zip(rows,outputs)
                    if row["kind"]==kind and bool(stats[row["topic"]][key])==present]
            n=len(chosen)
            out[(kind,"gold_in_source" if present else "gold_missing")]={
              "n":n,
              "strict":sum(x["strict"] for x in chosen),
              "unknown":sum(x["unknown"] for x in chosen),
              "mean_f1":round(sum(x["f1"] for x in chosen)/n,6) if n else 0.,
              "newline":sum(x["newline"] for x in chosen),
              "cap":sum(x["cap"] for x in chosen)}
    for kind,expected in (("bm25_answer_top1",3),("bm25_answer_top3",7)):
        if out[(kind,"gold_in_source")]["n"]!=expected or (
           out[(kind,"gold_missing")]["n"]!=PAPERS-expected):
            raise RuntimeError("HOLD altered retrieval coverage")
    return out

def report(by_mode,lengths,elapsed,retrieval,conditioned):
    if len(by_mode)!=2 or len(lengths)!=120 or len(conditioned)!=2:
        raise RuntimeError("HOLD incomplete real K2 output")
    stats=retrieval_audit(retrieval)
    if (stats["papers"],stats["answer_gold_top1"],
        stats["answer_gold_top3"],stats["answer_gold_any_index"])!=(12,3,7,12):
        raise RuntimeError("HOLD unexpected full paper retrieval statistics")
    lines=["# EXP-046 — Full-paper BM25 retrieval versus frozen K2 answer grounding",
      "",
      "**2026-10-15 | All 120 actual K2 Mid2 CPU free-generation calls; research only.**",
      "",
      "## Fixed scientific source and experiment custody",
      "",
      "- Official QASPER dev JSON SHA256: 2ae7ee62a65b1c4225791c70de80c2aad4e8998cf1fd4f09a53103db4f21af93.",
      "- Pre-inference frozen EXP046 protocol SHA256: 880f14d3fbfcddb322ac184a7525a95f3a0e8231cd4304bca21100b014a9c256.",
      "- Independent second 12-paper retrieval/prompt/gold cohort SHA256: f44ae4e5dcaf8f0b8455ee8f89f00207f8428461c5f536937ef11db01a2998f5.",
      "- Paper cohort is disjoint from EXP045 12-paper cohort, not publicly training-unseen.",
      "- Human oracle supporting evidence, BM25 top1/top3 for answerable query, BM25 top1/top3",
      "  for original human FULL-PAPER unanswerable question, 60 records per positional setting.",
      "- Same frozen Mid2 learned BF16 parameters, default RoPE and RL-derived YaRN rotary.",
      "- Greedy maximum 16 newly generated tokens; first decoded GENERATED newline stop.",
      "- BM25 ranks complete paper-indexed <=900-character text windows, but K2 sees only",
      "  selected top1/top3 windows, NOT the full original paper. This is retrieval-assisted",
      "  partial-context research, not end-to-end full-paper K2 ingestion.",
      "",
      "## Model-independent retriever coverage (answerable)",
      "",
      f"- Articles: {stats['papers']}. Index windows range: {stats['index_windows_min']}..{stats['index_windows_max']}.",
      f"- Human extractive gold string in any indexed window: {stats['answer_gold_any_index']}/12.",
      f"- Gold string retrieved in top1: {stats['answer_gold_top1']}/12.",
      f"- Gold string retrieved in top3: {stats['answer_gold_top3']}/12.",
      "- Exact literal-gold presence is a retrieval diagnostic; paraphrase/inference",
      "  could still support an answer where the exact gold string is missing.",
      "",
      "## Strict whole-answer and human unanswerable outcomes",
      "",
      "| RoPE | Case | N | Strict exact | First line exact | Mean token F1 | Exact UNKNOWN | First-newline | Native EOS | Token cap | Empty |",
      "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for mode in ("default","yarn"):
        for kind in KINDS:
            d=by_mode[mode][kind]
            lines.append(f"| {mode} | {kind} | {d['n']} | {d['strict']} | {d['first_line']} | {d['mean_f1']:.6f} | {d['unknown']} | {d['newline']} | {d['native_eos']} | {d['cap']} | {d['empty']} |")
    lines+=["",
      "## Retrieval hit-conditioned answerable results",
      "",
      "| RoPE | Retrieved setting | Literal gold in context | N | Strict correct | Mean token F1 | Exact UNKNOWN | Newline | Cap |",
      "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for mode in ("default","yarn"):
        for kind in ("bm25_answer_top1","bm25_answer_top3"):
            for g in ("gold_missing","gold_in_source"):
                d=conditioned[mode][(kind,g)]
                lines.append(f"| {mode} | {kind} | {g} | {d['n']} | {d['strict']} | {d['mean_f1']:.6f} | {d['unknown']} | {d['newline']} | {d['cap']} |")
    lines+=["", "## Scientific limits and safety", "",
      "- Gold presence/absence is a literal string check, NOT adjudication of semantic sufficiency.",
      "- Original QASPER human unanswerable labels apply to FULL papers, whereas this study",
      "  provides retrieved subsets of each paper; UNKNOWN correctness is a research proxy.",
      "- BM25 retrieval recall and generator comprehension failures are separate observations.",
      "- Public 12-paper clustered pilot, possible model training exposure, no independent blind holdout.",
      "- Greedy 16-new-token response formatting can reduce strict scores; token F1 is only descriptive.",
      "- No raw paper text, author/title/ID, model output, source QID, per-case predictions, logits,",
      "  activations, persistent checkpoint, or learned parameter changes were archived.",
      f"- Model input token min/max: {min(lengths)} / {max(lengths)}.",
      f"- Timed native generation elapsed: {elapsed:.2f} seconds.",
      "- Both PRs stay draft/unmerged. No training, model registry/serving/routing,",
      "  credential/network, production graph/NAS, deployment or arXiv submission.",
      ""]
    return "\n".join(lines)
