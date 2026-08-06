# Fleet capability routing deployment checklist

- [ ] `tailscale status --json` is readable by the controller user.
- [ ] The operator role policy has been reviewed and is mode `0600` or stricter.
- [ ] Unknown/unlisted peers remain `observer_only`.
- [ ] Exact-loadout comparison artifacts are current for model, quantization,
      runtime, context, cache, batching, and concurrency.
- [ ] `lms-fleet-routing-matrix` produces `fleet_routing_matrix.v1`.
- [ ] The artifact contains `admission.admitted: false`.
- [ ] `summary.tailnet_nodes` is greater than two.
- [ ] Phones and miscellaneous peers are visible but not routable.
- [ ] Auxiliary nodes have no `allow_agent_runtime` or `allow_code_execution`
      unless explicitly approved.
- [ ] AssistX import succeeds and reports the expected node/profile counts.
- [ ] AssistX context projection contains the complete tailnet census.
- [ ] Only separately admitted runtime/model records appear in the signed runtime
      projection.
- [ ] Summarization and compression can select auxiliary nodes.
- [ ] Coding rejects auxiliary-only nodes.
- [ ] A measured quality-floor failure cannot win solely on throughput.
- [ ] The refresh timer is enabled and its last run succeeded.
- [ ] Stale claim and expired projection negative canaries still pass.
