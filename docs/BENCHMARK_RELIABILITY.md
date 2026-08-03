# Benchmark reliability contract

The benchmark system is designed to answer a stricter question than “did this model respond once?” It must determine whether one exact runtime and model produce repeatable, complete, protocol-correct results under a reviewed loadout.

A fast result is not selectable when it is incomplete, unstable, heavily retried, produced by a different model, missing raw evidence, or resumed from evidence whose integrity no longer verifies.

## Measurement layers

A physical candidate passes through these layers:

1. **Exact model identity** — `/v1/models` must expose the planned model ID exactly. The runner never substitutes the first available model.
2. **Cold-load canary** — the endpoint must complete a deterministic `READY` request before measurement.
3. **Warmup stabilization** — three successful warmup requests are required by default. Their wall-time coefficient of variation must not exceed the configured threshold.
4. **Strict protocol measurement** — streaming responses must use event-stream content, contain valid JSON chunks, produce text, and terminate with `[DONE]`. Non-streaming responses must contain a valid textual choice.
5. **Complete trial** — one trial must contain every expected endpoint, case, and repeat key exactly once. Duplicate, missing, or unexpected samples invalidate the entire trial.
6. **Raw output verification** — every successful CSV row must reference one unique, non-empty output file inside the trial sidecar directory. The file size and SHA-256 are retained in the trial manifest.
7. **Post-trial health** — the exact model must remain exposed and pass another deterministic canary.
8. **Repeated trials** — three valid complete trials are required by default.
9. **Reliability aggregation** — the runner computes confidence, dispersion, completeness, success, and retry measurements before a candidate can be selected.
10. **Release verification** — `lms-fleet-gate` recomputes the standalone reliability fingerprint and matches it to the selected loadout.

## Whole-trial retries only

The runner does not retry individual failed prompts and then keep only their successful replacements. That would bias the evidence toward success and hide intermittent endpoint behavior.

A failed or incomplete trial remains preserved as diagnostic evidence. The system may start a new complete trial attempt when a process times out, required artifacts are incomplete, a raw output is missing, endpoint health changes, or the trial runner fails. The retry rate is itself a hard reliability metric.

Defaults:

```text
requested valid trials: 3
minimum valid trials:   3
maximum trial attempts: 5
maximum retry rate:     0.25
```

One failed attempt followed by three successful attempts produces a retry rate of 0.25 and is reviewable. Two failed attempts out of five exceed the default threshold and make the candidate ineligible.

## Deterministic order variation

Each trial uses a deterministic seed derived from the benchmark seed, trial number, and attempt number. The suite case order and inventory order are shuffled reproducibly between trials.

This reduces fixed-order bias from:

- cold versus warm cache position;
- thermal accumulation;
- allocator history;
- first-request initialization;
- systematic placement of long-context cases.

The input fingerprint binds the inventory, suite bytes, expanded case keys, benchmark parameters, reliability thresholds, and seed.

## Required statistics

The reliability report includes:

- sample completeness;
- request success rate;
- evaluator success rate;
- 95% Wilson lower confidence bound for request success;
- median TPS and TTFT;
- TPS p10 and p90;
- TTFT p90;
- per-trial median TPS and TTFT coefficient of variation;
- relative median absolute deviation for TPS and TTFT;
- deterministic bootstrap 95% confidence intervals for median TPS and TTFT;
- valid trial count, attempt count, and retry rate;
- warmup stability;
- a composite reliability score;
- explicit hard-failure reasons.

No outlier is silently deleted. Median, percentiles, MAD, and bootstrap intervals reduce sensitivity to individual extremes while preserving every raw sample.

## Default hard gates

A candidate is rejected when any default condition is violated:

```text
valid trials                         < 3
sample completeness                  < 1.00
request success rate                 < 0.98
evaluator success rate               < 0.90
95% Wilson success lower bound       < 0.80
trial TPS coefficient of variation   > 0.20
trial TTFT coefficient of variation  > 0.35
TPS relative MAD                     > 0.25
TTFT relative MAD                    > 0.25
trial retry rate                     > 0.25
warmup stability                     failed
post-trial exact-model health        failed
strict protocol validation           failed
raw output evidence                  missing, empty, duplicate, or unsafe
```

The existing execution gates still apply: completion, streaming, concurrency, memory headroom, sustained stability, no crash, and successful benchmark process exit.

## Evidence layout

Each candidate contains:

```text
benchmark/<candidate>/
  inventory.csv
  launch.json or mapped-endpoint evidence
  suite_command.json
  suite.log
  suite/
    config.json
    preflight.json
    reliability.json
    run_results.csv
    run_summary.csv
    task_summary.csv
    trials/
      trial_001/attempt_001/
        command.json
        runner.log
        trial_manifest.json
        inventory.csv
        suite.json
        output/
        sidecars/
```

Trial manifests record command, deterministic seed, timeout, return code, environment snapshots, post-trial health, aggregate artifact hashes, each successful sample output’s size and SHA-256, and failure classification. The manifest is sealed with a canonical `trial_manifest_fingerprint`. Failed attempts remain in the bundle.

## Reliability fingerprint

`reliability.json` contains a canonical `reliability_fingerprint`. The selected benchmark row carries the same fingerprint.

The sweep release gate:

- locates the selected candidate’s reliability report;
- recomputes the fingerprint;
- requires `passed=true` and `admission.admitted=false`;
- requires at least three valid trials;
- checks trial-count consistency;
- requires exactly one candidate summary;
- matches the model ID;
- matches completeness, dispersion, confidence, retry, warmup, and score fields to the selected row;
- rejects any reported reliability failures.

This prevents a CSV row or selected-loadout file from claiming reliability that the underlying report does not prove.

## Bounded execution

The reliability orchestrator calculates a conservative per-trial timeout from request timeout, case count, repeat count, and endpoint count. The fleet wrapper also applies an outer process-group timeout through `LMS_BENCH_SUITE_TIMEOUT`, defaulting to 6,900 seconds. The remote rollout remains bounded separately.

On outer timeout, the complete benchmark process group is terminated, then killed if necessary. The candidate receives a nonzero benchmark exit and cannot be selected.

## Resuming interrupted work

Pass `--resume` to reuse a previously completed trial only when all of these conditions hold:

- the input fingerprint matches exactly;
- the trial succeeded on its first attempt;
- there are no earlier failed attempts in that trial directory;
- the trial manifest fingerprint recomputes;
- the runner log hash matches;
- all aggregate artifact sizes and hashes match;
- every successful raw output still exists, is non-empty, and matches its retained size and SHA-256;
- post-trial endpoint health passed.

A trial with earlier failed attempts is rerun rather than resumed because reusing only the successful attempt would hide its retry rate from the aggregate. Unsealed legacy manifests are not resumable.

Resume is intended for controller or node interruption recovery. It is not a mechanism for replacing failed samples inside a trial.

## Admission boundary

A passing reliability report proves that the benchmark measurement is complete and repeatable under the recorded conditions. It does not install, start, expose, route, or admit the runtime. Physical identity, live model identity, private-path reachability, shared capacity, evidence freshness, and rollback remain external admission gates.
