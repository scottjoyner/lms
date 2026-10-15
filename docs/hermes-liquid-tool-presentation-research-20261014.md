# Liquid 24B native Hermes: tool-contract and A/B wire preflight

**2026-10-14 — research only, draft / NO MERGE / NO PROMOTION.**

Source baseline: `scottjoyner/lms` `82f18679eebad18d050a74914b616232a3af6b36`.
Earlier failed witness: [knowledge PR #136](https://github.com/scottjoyner/knowledge/pull/136).
Independent owner gate: [LMS PR #18](https://github.com/scottjoyner/lms/pull/18).

## Narrow changes
- Live disposable `graph_query` now **advertises** `read_only` as required and
  describes the mandatory `read_only=true` and current parameterization denial.
  The actual read-only gate and graph custody checks remain unchanged.
- MCP `tools/list` and argument validation both resolve the same instance-scoped
  tool catalog. Historical canned benchmarks retain their existing schemas.
- `hermes_wire_pair_preflight` evaluates metadata-only bridge (3 tools) versus
  direct (9 tools) captures and rejects any mismatch of first system/user digest,
  model ID, generation settings, or unexpected tool catalogs. Within-arm drift
  also fails closed. A pass means only *wire comparison admissible*, never a
  verified owner answer, model admission, deployment approval, or promotion.

## Reproduction
Use a fresh scratch research container, never production Neo4j.
Record outbound **metadata only**, with no raw prompts, headers or credentials.
Both trial arms must use the **same patched, pinned** LMS/Hermes source,
GGUF hash, fixed user task, Neo4j fixture, chat template, and parameters.

Keep `temperature=0`, `max_tokens=640`, `stream=true`, and bounded
135-second trial wall clock; establish limits and exact hash before starting.
Expose three bridge tools (A), then nine direct MCP tools (B); preserve the
native system preamble, and instrument each outgoing request.

    PYTHONPATH=src python3 -m lms_agent_bench.hermes_wire_pair_preflight \
      --bridge /path/to/bridge-native-metadata.jsonl \
      --direct /path/to/direct-native-metadata.jsonl

The preflight exits 3 and prints a denial if preambles or any required
wire controls differ. It is not a runner and does not start a model,
create a scratch graph, or approve a release.

## Follow-on acceptance (not established here)
Require evidence chain request -> model tool call -> authenticated scratch
query -> returned owner row -> tool response written to Hermes transport ->
completed source-grounded answer. A ledger from an offline query replay alone
does not prove that the model received that row.

Three **fresh** native trials must each finish correctly without timeout and
with zero prohibited effects, checked per trial against isolated real Neo4j.
Preserve failures; never silently retry or substitute archived canned passes.
Then review the full eight-case suite and independent exact-head CI.
Research success still does not authorize merge, model promotion, fleet routing,
production graph access, network/config/credential changes, or deployment.
