#!/usr/bin/env python3
"""Build/verify a NON-AUTHORIZING, source-free custodian handoff ZIP locally.

No external send, signatures, key enrollment, deployments, service or inference.
An SHA256 manifest is not independent trust, and all outcomes remain NO-GO.
"""

import argparse
import hashlib
import io
import json
import os
import pwd
import stat
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCHEMA = "exp053-custodian-handoff-packet-v1"
MANIFEST = "manifest.json"
MAX_INPUT = 16384
MAX_PACKET = 131072
# These exact reviewed public-safe files are explicitly allowlisted and pinned.
# Any source change requires deliberate code review and a separate source pin.
PINS = {
    "EXP053-INDEPENDENT-CUSTODIAN-REQUEST-2026-10-19.md":
        "03e1b46f548cf418926668c3ab19155bf42fe1af8717b8fc1d2d25f5626f70e6",
    "EXP053-CUSTODY-BOUNDARY-NEGATIVE-AUDIT-SPEC-2026-10-18.md":
        "46c06eab83811ea6028dab262a6bd1e7b127982b7ef70c17084938d29314b54a",
    "EXP053-INDEPENDENT-CUSTODY-NEGATIVE-CHALLENGES-SPEC-2026-10-18.md":
        "d3308a8b627c513e5b8aff200da2befaaafc625e33a992ca7ad32a095e007744",
    "EXP053-EXTERNAL-HIGHWATER-WITNESS-SPEC-2026-10-17.md":
        "3aa0547ee07a9c271fd67b6542544a147790fef052337f1ce1764a8c2cbb5f1e",
    "EXP053-OBSERVER-PUBLICATION-CHAIN-SPEC-2026-10-18.md":
        "a387aa42f93b55a8ad1325dbddd5621e906bfc165b23967e42f65e9ebceff5e5",
    "EXP053-INDEPENDENT-CUSTODY-CHALLENGE-HANDOFF-2026-10-18.md":
        "ac7b32178b9811ee43fc06f06da764afa6b392e6138223b57d35521770a429c9",
    "EXP053-LENOVO-CUSTODY-CONTROL-CLAIMS-2026-10-18.json":
        "b92fd4594eded89ed89f3bd03548d7c76174d0e59feca3571b5881a7a0935044",
}


class Hold(ValueError):
    """Source-free refusal (never include raw paths or user-controlled text)."""


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _local_regular_bytes(path, limit):
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        fd = os.open(path, flags)
        with os.fdopen(fd, "rb") as handle:
            if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                raise Hold("file is not regular")
            payload = handle.read(limit + 1)
    except OSError as exc:
        raise Hold("unable to safely read source or packet") from exc
    if not 1 <= len(payload) <= limit:
        raise Hold("file is empty or exceeds bound")
    return payload


def inventory(*, source_dir=ROOT):
    source = Path(source_dir)
    if source.is_symlink() or not source.is_dir():
        raise Hold("source root missing or symlinked")
    contents = {}
    for name, expected in PINS.items():
        payload = _local_regular_bytes(source / name, MAX_INPUT)
        if _sha(payload) != expected:
            raise Hold("allowlisted source digest differs from reviewed pin")
        contents[f"docs/{name}"] = payload
    return contents


def _manifest(contents):
    files = {name: {"sha256": _sha(payload), "bytes": len(payload)}
             for name, payload in contents.items()}
    packet_id = _sha(_canonical({"schema": SCHEMA, "files": files}))[:20]
    return {
        "schema": SCHEMA, "packet_id": packet_id,
        "decision": "NO_GO_RESEARCH_ONLY",
        "scientific_execution_authorized": False,
        "independent_administration_approved": False,
        "witness_durability_approved": False,
        "provenance": "CALLER_ASSEMBLED_UNTRUSTED_SOURCE_ONLY",
        "files": files,
    }


def render_packet(contents):
    if set(contents) != {f"docs/{name}" for name in PINS}:
        raise Hold("unexpected custodian packet inventory")
    for name, payload in contents.items():
        if (not isinstance(payload, bytes) or not 1 <= len(payload) <= MAX_INPUT or
                _sha(payload) != PINS[name.removeprefix("docs/")]):
            raise Hold("packet member differs from allowlisted source")
    items = {**contents, MANIFEST: _canonical(_manifest(contents))}
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED, allowZip64=False) as archive:
        for name in sorted(items):
            meta = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            meta.create_system = 3
            meta.external_attr = 0o100644 << 16
            meta.compress_type = zipfile.ZIP_STORED
            archive.writestr(meta, items[name])
    raw = stream.getvalue()
    if len(raw) > MAX_PACKET:
        raise Hold("rendered packet exceeds allowed size")
    return raw


def verify_packet(raw):
    if not isinstance(raw, bytes) or not 1 <= len(raw) <= MAX_PACKET:
        raise Hold("oversized or malformed custodian packet")
    try:
        with zipfile.ZipFile(io.BytesIO(raw), "r", allowZip64=False) as archive:
            infos = archive.infolist()
            expected_names = sorted([MANIFEST, *(f"docs/{name}" for name in PINS)])
            if (len(infos) != len(expected_names) or
                    [info.filename for info in infos] != expected_names):
                raise Hold("unexpected ZIP entries, ordering or duplicates")
            items = {}
            for info in infos:
                if (info.is_dir() or info.file_size > MAX_INPUT or
                        info.compress_type != zipfile.ZIP_STORED or
                        info.date_time != (1980, 1, 1, 0, 0, 0) or
                        info.external_attr != (0o100644 << 16) or
                        info.create_system != 3):
                    raise Hold("unreviewed ZIP entry attributes")
                payload = archive.read(info)
                if len(payload) != info.file_size:
                    raise Hold("ZIP entry size mismatch")
                items[info.filename] = payload
    except (OSError, RuntimeError, ValueError, zipfile.BadZipFile, EOFError) as exc:
        raise Hold("invalid or tampered ZIP source") from exc
    try:
        manifest = json.loads(items[MANIFEST])
    except (ValueError, UnicodeError, TypeError, RecursionError) as exc:
        raise Hold("invalid or deeply nested packet manifest") from exc
    contents = {name: data for name, data in items.items() if name != MANIFEST}
    if manifest != _manifest(contents):
        raise Hold("packet manifest mismatch")
    if render_packet(contents) != raw:
        raise Hold("packet bytes are not the deterministic reviewed rendering")
    return {
        "schema": SCHEMA,
        "packet_id": manifest["packet_id"],
        "packet_sha256": _sha(raw),
        "document_count": len(contents),
        "decision": "NO_GO_RESEARCH_ONLY",
        "scientific_execution_authorized": False,
        "independent_administration_approved": False,
        "witness_durability_approved": False,
        "provenance": "CALLER_ASSEMBLED_UNTRUSTED_SOURCE_ONLY",
    }


def _account_home():
    return Path(pwd.getpwuid(os.getuid()).pw_dir)


def _safe_output(path):
    target = Path(path)
    home = _account_home()
    if not target.is_absolute() or ".." in target.parts or not target.is_relative_to(home):
        raise Hold("packet output must be an absolute user-local path")
    parent = target.parent
    if parent.is_symlink():
        raise Hold("symlinked packet output parent")
    if not parent.is_dir() or parent == home:
        raise Hold("packet output requires an existing user-owned subfolder")
    step = parent
    while step != home:
        if step.is_symlink():
            raise Hold("symlinked packet output parent")
        step = step.parent
    if parent.stat().st_uid != os.getuid() or parent.stat().st_mode & 0o022:
        raise Hold("packet output parent not owner-private")
    return target


def create(path, *, source_dir=ROOT):
    target = _safe_output(path)
    raw = render_packet(inventory(source_dir=source_dir))
    report = verify_packet(raw)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(target, flags, 0o600)
    except OSError as exc:
        raise Hold("destination exists or cannot be safely created") from exc
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        if _local_regular_bytes(target, MAX_PACKET) != raw:
            raise Hold("newly written packet failed exact-byte verification")
    except Exception:
        target.unlink(missing_ok=True)
        raise
    return {**report, "status": "CREATED_USER_LOCAL_ONLY"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--inspect", action="store_true",
                        help="read-only check of reviewed source documents")
    choice.add_argument("--create", type=Path,
                        help="explicit local packet path, no overwrite/no remote send")
    choice.add_argument("--verify", type=Path,
                        help="read-only verify an existing local packet")
    args = parser.parse_args(argv)
    try:
        if args.inspect:
            packet = render_packet(inventory())
            report = {**verify_packet(packet), "status": "NO_FILES_WRITTEN"}
        elif args.create is not None:
            report = create(args.create)
        else:
            report = {**verify_packet(_local_regular_bytes(args.verify, MAX_PACKET)),
                      "status": "LOCAL_PACKET_BYTES_VERIFIED_ONLY"}
    except (Hold, OSError, ValueError):
        print('{"decision":"NO_GO_RESEARCH_ONLY","scientific_execution_authorized":false,'
              '"independent_administration_approved":false,'
              '"witness_durability_approved":false,"status":"INVALID_OR_UNVERIFIABLE_PACKET"}')
        return 2
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 2  # Even an intact packet is NOT authority to deploy or infer.


if __name__ == "__main__":
    raise SystemExit(main())
