"""Signed runtime identity witness binding qualified model bytes to one live process."""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence
from urllib.parse import urlparse, urlunparse

from lms_agent_bench import fleet_evidence_attestation as _ssh
from lms_agent_bench import fleet_operator as _operator
from lms_agent_bench import runtime_canary_attestation as _canary_attestation
from lms_agent_bench.model_loadout import validate_manifest

SCHEMA_VERSION = "fleet-runtime-identity-witness.v1"
DEFAULT_NAMESPACE = "lms-runtime-identity-witness"
CONTINUITY_SCHEMA_VERSION = "fleet-runtime-continuity-attestation.v1"
CONTINUITY_NAMESPACE = "lms-runtime-continuity"


def _normalize_runtime_url(value: str) -> str:
    raw = str(value or "").strip()
    if "://" not in raw:
        raw = "http://" + raw
    parsed = urlparse(raw)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("runtime URL must be a valid http(s) endpoint")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("runtime URL may not contain credentials")
    path = parsed.path.rstrip("/")
    if path == "/v1":
        path = ""
    return urlunparse((parsed.scheme, parsed.netloc, path, "", "", "")).rstrip("/")


def _canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(payload),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")


def _platform_name() -> str:
    return platform.system().strip().lower()


def _run_text(args: Sequence[str], label: str, *, timeout: int = 15) -> str:
    process = subprocess.run(
        list(args),
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    if process.returncode != 0:
        detail = process.stderr.strip() or process.stdout.strip()
        raise ValueError(f"{label} failed: {detail}" if detail else f"{label} failed")
    return process.stdout


def _boot_id() -> str:
    if _platform_name() == "darwin":
        output = _run_text(
            ["/usr/sbin/sysctl", "-n", "kern.boottime"],
            "Darwin boot identity query",
        )
        match = re.search(r"sec\s*=\s*(\d+)\s*,\s*usec\s*=\s*(\d+)", output)
        if not match:
            raise ValueError("unable to parse Darwin boot identity")
        return f"darwin-boottime:{match.group(1)}.{match.group(2).zfill(6)}"

    value = Path("/proc/sys/kernel/random/boot_id").read_text(
        encoding="utf-8"
    ).strip()
    if not value:
        raise ValueError("boot identity is unavailable")
    return value


def _process_start_ticks(pid: int) -> int:
    if _platform_name() == "darwin":
        output = _run_text(
            ["/bin/ps", "-p", str(pid), "-o", "lstart="],
            f"Darwin process start query for pid {pid}",
        ).strip()
        if not output:
            raise ValueError(f"unable to observe process start time for pid {pid}")
        try:
            started = datetime.strptime(output, "%a %b %d %H:%M:%S %Y")
        except ValueError as exc:
            raise ValueError(
                f"unable to parse process start time for pid {pid}"
            ) from exc
        return int(time.mktime(started.timetuple()))

    stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    try:
        tail = stat.rsplit(")", 1)[1].strip().split()
        return int(tail[19])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"unable to parse process start time for pid {pid}") from exc


def _stat_identity(stat: os.stat_result) -> dict[str, int]:
    return {
        "device": int(stat.st_dev),
        "inode": int(stat.st_ino),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "ctime_ns": int(stat.st_ctime_ns),
    }


def _same_file_identity(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    try:
        return all(
            int(left.get(key)) == int(right.get(key))
            for key in ("device", "inode", "size_bytes", "mtime_ns", "ctime_ns")
        )
    except (TypeError, ValueError):
        return False


def _stable_file_sha256(path: Path, label: str) -> tuple[str, dict[str, int]]:
    before = _stat_identity(path.stat())
    digest = _ssh.file_sha256(path)
    after = _stat_identity(path.stat())
    if not _same_file_identity(before, after):
        raise ValueError(f"{label} changed while it was being hashed")
    return digest, after


def _darwin_lsof_records(pid: int, *, descriptors: str | None = None) -> list[dict[str, str]]:
    args = ["/usr/sbin/lsof", "-nP", "-a", "-p", str(pid)]
    if descriptors:
        args.extend(["-d", descriptors])
    args.extend(["-F", "fDin"])
    output = _run_text(args, f"Darwin lsof query for pid {pid}")
    records: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    for raw in output.splitlines():
        if not raw:
            continue
        field = raw[0]
        value = raw[1:]
        if field == "f":
            if current is not None:
                records.append(current)
            current = {"f": value}
        elif current is not None and field in {"D", "i", "n"}:
            current[field] = value
    if current is not None:
        records.append(current)
    return records


def _darwin_record_matches_identity(
    record: Mapping[str, str],
    file_identity: Mapping[str, Any],
) -> bool:
    try:
        expected_device = int(file_identity.get("device"))
        expected_inode = int(file_identity.get("inode"))
        observed_inode = int(record.get("i") or 0)
        device_text = str(record.get("D") or "")
        observed_device = int(device_text, 16) if device_text.startswith("0x") else int(device_text)
    except (TypeError, ValueError):
        return False
    return observed_device == expected_device and observed_inode == expected_inode


def _process_executable_path(pid: int) -> Path:
    if _platform_name() == "darwin":
        for record in _darwin_lsof_records(pid, descriptors="txt"):
            raw_path = str(record.get("n") or "")
            if not raw_path.startswith("/"):
                continue
            candidate = Path(raw_path)
            try:
                if candidate.is_file():
                    return candidate
            except OSError:
                continue
        raise ValueError(f"unable to resolve executable text file for pid {pid}")

    path = Path(f"/proc/{pid}/exe")
    try:
        path.stat()
    except OSError as exc:
        raise ValueError(f"unable to inspect executable for pid {pid}") from exc
    return path


def _process_executable_basename(pid: int) -> str:
    if _platform_name() == "darwin":
        return _process_executable_path(pid).name

    try:
        target = os.readlink(f"/proc/{pid}/exe")
    except OSError as exc:
        raise ValueError(f"unable to resolve executable for pid {pid}") from exc
    target = target.removesuffix(" (deleted)")
    name = Path(target).name
    if not name:
        raise ValueError(f"unable to determine executable basename for pid {pid}")
    return name


def _darwin_vmmap_contains_path(pid: int, path: Path) -> bool:
    try:
        output = _run_text(
            ["/usr/bin/vmmap", "-w", str(pid)],
            f"Darwin vmmap query for pid {pid}",
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError, ValueError):
        return False
    target = str(path)
    resolved = str(path.resolve())
    return target in output or resolved in output


def _mapped_file_binding(
    pid: int,
    file_identity: Mapping[str, Any],
    file_path: Path | None = None,
) -> bool:
    try:
        expected_dev = int(file_identity.get("device"))
        expected_inode = int(file_identity.get("inode"))
    except (TypeError, ValueError):
        return False
    if expected_inode <= 0:
        return False

    if _platform_name() == "darwin":
        if file_path is None or not _darwin_vmmap_contains_path(pid, file_path):
            return False
        return any(
            _darwin_record_matches_identity(record, file_identity)
            for record in _darwin_lsof_records(pid)
        )

    expected_major = os.major(expected_dev)
    expected_minor = os.minor(expected_dev)
    try:
        lines = Path(f"/proc/{pid}/maps").read_text(
            encoding="utf-8",
            errors="replace",
        ).splitlines()
    except OSError:
        return False
    for line in lines:
        fields = line.split(maxsplit=5)
        if len(fields) < 5:
            continue
        device = fields[3]
        inode = fields[4]
        try:
            major_text, minor_text = device.split(":", 1)
            major = int(major_text, 16)
            minor = int(minor_text, 16)
            mapped_inode = int(inode)
        except (ValueError, TypeError):
            continue
        if (
            major == expected_major
            and minor == expected_minor
            and mapped_inode == expected_inode
        ):
            return True
    return False


def _process_command_line(pid: int) -> str:
    if _platform_name() == "darwin":
        try:
            return _run_text(
                ["/bin/ps", "-p", str(pid), "-o", "command="],
                f"Darwin command-line query for pid {pid}",
            ).strip()
        except (OSError, subprocess.SubprocessError, ValueError):
            return ""
    try:
        return (
            Path(f"/proc/{pid}/cmdline")
            .read_bytes()
            .replace(b"\x00", b" ")
            .decode("utf-8", errors="replace")
        )
    except OSError:
        return ""


def _process_references_model(
    pid: int,
    model_path: Path,
    model_identity: Mapping[str, Any],
) -> str:
    # Strong evidence requires stable device+inode identity plus a kernel/OS
    # virtual-memory view. Path-only evidence is diagnostic because a pathname
    # can be replaced while an older inode remains mapped.
    if _mapped_file_binding(pid, model_identity, model_path):
        return (
            "darwin_vmmap_lsof"
            if _platform_name() == "darwin"
            else "proc_maps"
        )

    cmdline = _process_command_line(pid)
    if str(model_path) in cmdline:
        return "cmdline"
    raise ValueError(
        "model file identity is not mapped by the selected process and its path "
        "is absent from cmdline"
    )


def _regular_model(path: Path) -> Path:
    value = Path(path).expanduser()
    if value.is_symlink():
        raise ValueError("model path may not be a symbolic link")
    resolved = value.resolve()
    if not resolved.is_file():
        raise ValueError(f"model path is not a regular file: {resolved}")
    return resolved


def process_identity(pid: int) -> dict[str, Any]:
    if pid <= 0:
        raise ValueError("pid must be positive")
    executable = _process_executable_path(pid)
    executable_sha256, executable_file_identity = _stable_file_sha256(
        executable,
        "process executable",
    )
    platform_name = _platform_name()
    return {
        "pid": pid,
        "platform": platform_name,
        "boot_id": _boot_id(),
        "process_start_ticks": _process_start_ticks(pid),
        "process_start_source": (
            "darwin_ps_lstart_epoch_seconds"
            if platform_name == "darwin"
            else "linux_proc_start_ticks"
        ),
        "executable_sha256": executable_sha256,
        "executable_basename": _process_executable_basename(pid),
        "executable_file_identity": executable_file_identity,
    }


def build_witness(
    *,
    loadout_raw: Mapping[str, Any],
    canary_run_dir: Path,
    allowed_signers: Path,
    canary_identity: str,
    pid: int,
    runtime_url: str,
    runtime_kind: str,
    provider_model: str,
    model_path: Path,
    signing_key: Path,
    witness_identity: str,
    namespace: str = DEFAULT_NAMESPACE,
) -> tuple[dict[str, Any], bytes]:
    loadout = validate_manifest(loadout_raw, require_fingerprint=True)
    canary = _canary_attestation.verify_attestation(
        canary_run_dir,
        allowed_signers,
        canary_identity,
        require_success=True,
    )
    if canary.get("loadout_fingerprint") != loadout["loadout_fingerprint"]:
        raise ValueError("runtime canary belongs to a different exact loadout")

    model = _regular_model(model_path)
    actual_model_sha, model_file_identity = _stable_file_sha256(
        model,
        "live model file",
    )
    if actual_model_sha != loadout["model"]["content_sha256"]:
        raise ValueError("live model file does not match loadout model.content_sha256")

    process = process_identity(pid)
    binding_method = _process_references_model(pid, model, model_file_identity)
    normalized_url = _normalize_runtime_url(runtime_url)
    runtime_kind = str(runtime_kind or "").strip().lower()
    provider_model = str(provider_model or "").strip()
    if not runtime_kind or not provider_model:
        raise ValueError("runtime kind and provider model are required")

    signing_key_path = _ssh._require_regular(signing_key, "witness signing key", private=True)  # noqa: SLF001
    witness_identity = _ssh._identity(witness_identity)  # noqa: SLF001
    namespace = _ssh._namespace(namespace)  # noqa: SLF001
    core = {
        "schema_version": SCHEMA_VERSION,
        "node_id": str(loadout["node_id"]),
        "runtime_url": normalized_url,
        "runtime_kind": runtime_kind,
        "provider_model": provider_model,
        "loadout_fingerprint": loadout["loadout_fingerprint"],
        "model_id": str(loadout["model"]["id"]),
        "model_content_sha256": loadout["model"]["content_sha256"],
        "model_size_bytes": int(model_file_identity["size_bytes"]),
        "model_path": str(model),
        "model_process_binding": binding_method,
        "model_file_identity": model_file_identity,
        "process": process,
        "canary": {
            "run_id": canary.get("run_id"),
            "canary_id": canary.get("canary_id"),
            "manifest_fingerprint": canary.get("manifest_fingerprint"),
            "attestation_fingerprint": canary.get("attestation_fingerprint"),
            "signing_key_fingerprint": canary.get("signing_key_fingerprint"),
            "rollback_succeeded": canary.get("rollback_succeeded") is True,
        },
        "witness_signer_identity": witness_identity,
        "witness_signature_namespace": namespace,
        "witness_signing_key_fingerprint": _ssh._key_fingerprint(signing_key_path),  # noqa: SLF001
        "admission": {"admitted": False},
        "created_at_unix": int(time.time()),
    }
    witness = {
        **core,
        "witness_fingerprint": _operator.canonical_hash(core),
    }
    return witness, _canonical_bytes(witness)


def _sign_payload_bytes(
    payload: bytes,
    *,
    signing_key: Path,
    namespace: str,
    label: str,
) -> bytes:
    key = _ssh._require_regular(signing_key, f"{label} signing key", private=True)  # noqa: SLF001
    namespace = _ssh._namespace(namespace)  # noqa: SLF001
    import tempfile

    with tempfile.TemporaryDirectory(prefix=f"{label}-") as directory:
        path = Path(directory) / "payload.json"
        path.write_bytes(payload)
        process = subprocess.run(
            [
                _ssh._ssh_keygen(),  # noqa: SLF001
                "-Y",
                "sign",
                "-f",
                str(key),
                "-n",
                namespace,
                str(path),
            ],
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
        if process.returncode != 0:
            raise ValueError(f"{label} signing failed: " + process.stderr.strip())
        signature = Path(str(path) + ".sig")
        if not signature.is_file() or signature.stat().st_size <= 0:
            raise ValueError(f"ssh-keygen produced no {label} signature")
        return signature.read_bytes()


def sign_witness_bytes(
    payload: bytes,
    *,
    signing_key: Path,
    namespace: str = DEFAULT_NAMESPACE,
) -> bytes:
    return _sign_payload_bytes(
        payload,
        signing_key=signing_key,
        namespace=namespace,
        label="runtime-witness",
    )


def verify_witness_bytes(
    payload: bytes,
    signature: bytes,
    *,
    allowed_signers: Path,
    identity: str,
    namespace: str = DEFAULT_NAMESPACE,
) -> dict[str, Any]:
    signers = _ssh._require_regular(allowed_signers, "allowed signers")  # noqa: SLF001
    identity = _ssh._identity(identity)  # noqa: SLF001
    namespace = _ssh._namespace(namespace)  # noqa: SLF001
    try:
        witness = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("runtime witness payload is not valid JSON") from exc
    if not isinstance(witness, dict) or witness.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported runtime identity witness schema")
    if witness.get("admission") != {"admitted": False}:
        raise ValueError("runtime identity witness must remain non-admitted")
    fingerprint = str(witness.get("witness_fingerprint") or "")
    core = {key: value for key, value in witness.items() if key != "witness_fingerprint"}
    if fingerprint != _operator.canonical_hash(core):
        raise ValueError("runtime identity witness fingerprint mismatch")
    if _canonical_bytes(witness) != payload:
        raise ValueError("runtime identity witness must use canonical JSON encoding")

    import tempfile

    with tempfile.TemporaryDirectory(prefix="runtime-witness-verify-") as directory:
        signature_path = Path(directory) / "witness.sig"
        signature_path.write_bytes(signature)
        process = subprocess.run(
            [
                _ssh._ssh_keygen(),  # noqa: SLF001
                "-Y",
                "verify",
                "-f",
                str(signers),
                "-I",
                identity,
                "-n",
                namespace,
                "-s",
                str(signature_path),
            ],
            input=payload,
            capture_output=True,
            timeout=30,
            check=False,
        )
    if process.returncode != 0:
        stderr = process.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError("runtime witness signature verification failed: " + stderr)
    return witness


def observe_process_continuity(witness: Mapping[str, Any]) -> dict[str, Any]:
    process = witness.get("process")
    if not isinstance(process, Mapping):
        return {
            "valid": False,
            "reason": "witness_process_missing",
            "checked_at": int(time.time()),
        }
    try:
        pid = int(process.get("pid") or 0)
        current_boot_id = _boot_id()
        current_start_ticks = _process_start_ticks(pid)
        current_executable_basename = _process_executable_basename(pid)
        current_executable_identity = _stat_identity(
            _process_executable_path(pid).stat()
        )
    except (OSError, TypeError, ValueError):
        return {
            "valid": False,
            "reason": "process_not_observable",
            "checked_at": int(time.time()),
        }

    expected_executable_identity = process.get("executable_file_identity")
    process_valid = (
        pid > 0
        and current_boot_id == str(process.get("boot_id") or "")
        and current_start_ticks == int(process.get("process_start_ticks") or 0)
        and current_executable_basename
        == str(process.get("executable_basename") or "")
        and isinstance(expected_executable_identity, Mapping)
        and _same_file_identity(
            expected_executable_identity,
            current_executable_identity,
        )
    )

    model_identity = witness.get("model_file_identity")
    model_path = Path(str(witness.get("model_path") or ""))
    model_binding: str | None = None
    model_file_valid = False
    try:
        if isinstance(model_identity, Mapping):
            current_model = _regular_model(model_path)
            current_model_identity = _stat_identity(current_model.stat())
            model_file_valid = _same_file_identity(
                model_identity,
                current_model_identity,
            )
            if model_file_valid:
                model_binding = _process_references_model(
                    pid,
                    current_model,
                    current_model_identity,
                )
    except (OSError, TypeError, ValueError):
        model_file_valid = False
        model_binding = None

    model_binding_valid = model_binding is not None
    valid = process_valid and model_file_valid and model_binding_valid
    if valid:
        reason = "match"
    elif not process_valid:
        reason = "process_identity_changed"
    elif not model_file_valid:
        reason = "model_file_identity_changed"
    else:
        reason = "model_process_binding_changed"
    return {
        "valid": valid,
        "reason": reason,
        "checked_at": int(time.time()),
        "pid": pid,
        "boot_id": current_boot_id,
        "process_start_ticks": current_start_ticks,
        "executable_basename": current_executable_basename,
        "executable_file_valid": (
            isinstance(expected_executable_identity, Mapping)
            and _same_file_identity(
                expected_executable_identity,
                current_executable_identity,
            )
        ),
        "model_file_valid": model_file_valid,
        "model_process_binding_valid": model_binding_valid,
        "model_process_binding": model_binding,
    }



def build_continuity_attestation(
    witness: Mapping[str, Any],
    *,
    runtime_observation: Mapping[str, Any],
    signing_key: Path,
    signer_identity: str,
    namespace: str = CONTINUITY_NAMESPACE,
) -> tuple[dict[str, Any], bytes, bytes]:
    if witness.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("continuity attestation requires a runtime identity witness")
    witness_fingerprint = str(witness.get("witness_fingerprint") or "")
    if not witness_fingerprint.startswith("sha256:"):
        raise ValueError("runtime witness fingerprint is missing")
    observation_id = str(
        runtime_observation.get("runtime_observation_id") or ""
    ).strip()
    if not observation_id:
        raise ValueError("runtime observation ID is required")
    observed_models = sorted(
        {
            str(model).strip()
            for model in (runtime_observation.get("models") or [])
            if str(model).strip()
        },
        key=str.casefold,
    )
    try:
        observed_at = int(runtime_observation.get("observed_at") or 0)
    except (TypeError, ValueError) as exc:
        raise ValueError("runtime observation timestamp is invalid") from exc
    observation_evidence = {
        "runtime_observation_id": observation_id,
        "observed_at": observed_at,
        "runtime_kind": str(runtime_observation.get("runtime_kind") or ""),
        "protocol": str(runtime_observation.get("protocol") or ""),
        "base_url": str(runtime_observation.get("base_url") or "").rstrip("/"),
        "models": observed_models,
        "ready": bool(runtime_observation.get("ready")) and bool(observed_models),
        "observed_model_count": len(observed_models),
    }
    if not all(
        (
            observation_evidence["runtime_kind"],
            observation_evidence["protocol"],
            observation_evidence["base_url"],
            observed_at > 0,
        )
    ):
        raise ValueError("runtime observation evidence is incomplete")
    signer_identity = _ssh._identity(signer_identity)  # noqa: SLF001
    namespace = _ssh._namespace(namespace)  # noqa: SLF001
    if signer_identity != str(witness.get("node_id") or ""):
        raise ValueError("continuity signer identity must equal witness node_id")

    signing_key_path = _ssh._require_regular(  # noqa: SLF001
        signing_key,
        "runtime continuity signing key",
        private=True,
    )
    continuity = observe_process_continuity(witness)
    core = {
        "schema_version": CONTINUITY_SCHEMA_VERSION,
        "node_id": str(witness.get("node_id") or ""),
        "runtime_observation_id": observation_id,
        "observation": observation_evidence,
        "witness_fingerprint": witness_fingerprint,
        "runtime_url": str(witness.get("runtime_url") or ""),
        "runtime_kind": str(witness.get("runtime_kind") or ""),
        "provider_model": str(witness.get("provider_model") or ""),
        "continuity": continuity,
        "signer_identity": signer_identity,
        "signature_namespace": namespace,
        "signing_key_fingerprint": _ssh._key_fingerprint(signing_key_path),  # noqa: SLF001
        "admission": {"admitted": False},
    }
    attestation = {
        **core,
        "attestation_fingerprint": _operator.canonical_hash(core),
    }
    payload = _canonical_bytes(attestation)
    signature = _sign_payload_bytes(
        payload,
        signing_key=signing_key_path,
        namespace=namespace,
        label="runtime-continuity",
    )
    return attestation, payload, signature


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="lms-runtime-witness")
    parser.add_argument("--loadout", type=Path, required=True)
    parser.add_argument("--canary-run-dir", type=Path, required=True)
    parser.add_argument("--allowed-signers", type=Path, required=True)
    parser.add_argument("--canary-identity", required=True)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--runtime-url", required=True)
    parser.add_argument("--runtime-kind", required=True)
    parser.add_argument("--provider-model", required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--signing-key", type=Path, required=True)
    parser.add_argument("--witness-identity", required=True)
    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        loadout_raw = json.loads(args.loadout.read_text(encoding="utf-8"))
        if not isinstance(loadout_raw, dict):
            raise ValueError("loadout must be a JSON object")
        witness, payload = build_witness(
            loadout_raw=loadout_raw,
            canary_run_dir=args.canary_run_dir,
            allowed_signers=args.allowed_signers,
            canary_identity=args.canary_identity,
            pid=args.pid,
            runtime_url=args.runtime_url,
            runtime_kind=args.runtime_kind,
            provider_model=args.provider_model,
            model_path=args.model_path,
            signing_key=args.signing_key,
            witness_identity=args.witness_identity,
            namespace=args.namespace,
        )
        signature = sign_witness_bytes(
            payload,
            signing_key=args.signing_key,
            namespace=args.namespace,
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_bytes(payload)
        Path(str(args.out) + ".sig").write_bytes(signature)
        print(json.dumps(witness, indent=2, sort_keys=True))
        return 0
    except (OSError, ValueError, json.JSONDecodeError, subprocess.SubprocessError) as exc:
        print(f"runtime identity witness failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
