# CLI reconciliation — 2026-07-30

The package previously installed a console command named `lms`. That shadowed LM Studio's official `lms` CLI while the package's own Link-aware bridge attempted to call that official command.

On branch `full-auto-reconciliation-20260730`:

- the agent-facing convenience command is renamed to `lms-agent`;
- `lms-bench`, `lmsbench`, `lms-bench-endpoints`, and `lmstudio-bridge` remain available;
- the unqualified `lms` command is reserved for LM Studio;
- loaded-process discovery should use `lmstudio-bridge models --loaded-only --host <physical-host> --json`, which delegates to official `lms ps --json --host <physical-host>`;
- `/v1/models` remains an API-visibility probe and must not be treated as proof of the physical runtime owner.

## Upgrade

Reinstall the package after switching to this branch:

```bash
python3 -m pip uninstall -y lms-agent-bench
python3 -m pip install -e .
command -v lms
command -v lms-agent
lms --help
lms-agent doctor
```

`command -v lms` must resolve to LM Studio's official CLI, not this repository.
