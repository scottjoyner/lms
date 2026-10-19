# Offline LMS wheelhouse qualification (research/production hold)

The CLI `lms-offline-wheel-verify` is a **read-only integrity preflight**, not a
package installer, signed release attestation, source-provenance check, or
physical canary authority. It reads a wheel directory and a newline-separated
hash lock with exactly one `name==version --hash=sha256:<64 hex>` per wheel.
It rejects unpinned or duplicate distributions, missing wheels, symlinked or
group/world-writable inputs, mismatched hashes, unreadable wheels, and mismatched
embedded package metadata. It emits JSON with `production_authorized=false`.

## Reproducible staging flow (Python 3.12, x86_64 only)

On `x1-370`, the owner-only reference snapshot is
`/home/scott/.lms-canary-hermetic-qual-20261018`, built from exact PR #33
commit `0879c4b878f3e259eca1bcbb251cdf21de6bb987`. Its wheelhouse contains
LMS 0.33.0 and five frozen runtime dependencies; the SHA-256 lockfile is
`f843dec9158288f70813a9edda06afb6fbb70b33a316b948c472493808064da5`.
This pin covers the captured artifacts, not independently verified publisher
signatures or bit-for-bit rebuild reproducibility.

To verify the captured owner-controlled artifacts, invoke:

```bash
lms-offline-wheel-verify \
  --wheel-dir "$ROOT/wheels" \
  --lock "$ROOT/requirements-locked.sha256.txt" \
  --expected-version 0.33.0
```

Before installation, independently verify the trusted lockfile digest and the
source Git commit using a separately controlled trust channel. Then create a
fresh `python3 -m venv` with `include-system-site-packages = false` and install
only the preverified wheel files:

```bash
"$ROOT/isolated-venv/bin/python" -m pip install --no-index \
  --find-links "$ROOT/wheels" --require-hashes \
  -r "$ROOT/requirements-locked.sha256.txt"
"$ROOT/isolated-venv/bin/python" -m pip check
```

`ROOT` must name an explicitly selected, owner-protected staging directory.
Do not install to any active LM Studio server environment or the shared LMS
checkout. Testing on another Python ABI or architecture requires an independent
resolved and reviewed wheel set, including platform-specific wheels.

## Remaining NO-GO gates

Wheel hashes do not prove package safety, independent custody, supply-chain
origin, true build reproducibility, or semantic compatibility with production
models. No live candidate, signed physical rollback, stable-model approval,
independent observer keys, or profile admission follows from this test.
EXP-053 human adjudication and scientific authorization remain separate.
