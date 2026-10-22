#!/usr/bin/env python3
"""Install/verify an immutable, source-only EXP053 preflight CLI on this host.

NO daemon, no production service, model inference, witness key, SSH transfer,
fleet routing, NAS, or Neo4j writes. All deployment writes stay in explicit
owner-scoped user-local prefix; remote probes remain read-only.
"""
import argparse
import hashlib
import json
import os
import pwd
import shutil
import stat
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# A single immutable controller-only release includes both *non-authorizing*
# diagnostics. Neither program installs remote software or enables a witness.
FILES = (
    "exp053_witness_host_preflight.py",
    "EXP053-OFFHOST-READONLY-PREFLIGHT-SPEC-2026-10-17.md",
    "exp053_ssh_custody_overlap.py",
    "EXP053-CONTROLLER-SSH-CUSTODY-OVERLAP-HANDOFF-2026-10-18.md",
)
SCHEMA = "exp053-readonly-preflight-release-v1"
MANIFEST = "release-manifest.json"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def fail(message):
    raise ValueError("HOLD: preflight release " + message)


def _checked_file(path):
    if path.is_symlink() or not path.is_file() or not stat.S_ISREG(path.stat().st_mode):
        fail("source/release file is missing, invalid or symlinked")
    return path.read_bytes()


def source_inventory(source_dir=ROOT):
    source_dir = Path(source_dir)
    if source_dir.is_symlink() or not source_dir.is_dir():
        fail("source directory not an ordinary directory")
    files = {}
    for name in FILES:
        data = _checked_file(source_dir / name)
        if not data or len(data) > (1 << 20):
            fail("source file empty or unexpectedly large")
        files[name] = digest(data)
    return files


def release_id(files):
    return digest(json.dumps(files,sort_keys=True,separators=(",",":")).encode())[:20]


def _account_home():
    return Path(pwd.getpwuid(os.getuid()).pw_dir).resolve()


def _owner_prefix(prefix):
    candidate = Path(prefix).expanduser()
    if not candidate.is_absolute():
        fail("prefix must be an absolute local path")
    # Refuse all symlinked path segments. Do not install as root or outside
    # the caller's home; this tool is NOT a system-wide service installer.
    home = _account_home()
    resolved = candidate.resolve(strict=False)
    if resolved == home or not resolved.is_relative_to(home):
        fail("prefix must be a subdirectory of user's home")
    path = candidate
    while path != home:
        if path.is_symlink():
            fail("symlinked release prefix component")
        if path.parent == path:
            fail("release prefix escaped home")
        path = path.parent
    return candidate


def verify(release_dir, *, expected=None):
    release = Path(release_dir)
    if release.is_symlink() or not release.is_dir():
        fail("installed release directory missing/unsafe")
    raw = _checked_file(release / MANIFEST)
    try:
        manifest = json.loads(raw)
    except (UnicodeDecodeError,ValueError):
        fail("release manifest is malformed")
    if (not isinstance(manifest,dict) or set(manifest) !=
            {"schema","release_id","files"} or manifest["schema"] != SCHEMA or
            not isinstance(manifest["files"],dict) or set(manifest["files"]) != set(FILES)):
        fail("release manifest schema/contents drift")
    observed = {name:digest(_checked_file(release/name)) for name in FILES}
    if manifest["files"] != observed or manifest["release_id"] != release_id(observed):
        fail("release has been tampered with")
    if expected is not None and manifest["files"] != expected:
        fail("installed release differs from reviewed local files")
    if set(os.listdir(release)) != {*FILES,MANIFEST}:
        fail("unexpected file present in release")
    return {"schema":SCHEMA,"release_id":manifest["release_id"],
            "files":observed,"verified":True}


def install(prefix, *, source_dir=ROOT):
    target = _owner_prefix(prefix)
    files = source_inventory(source_dir)
    rid = release_id(files)
    target.mkdir(parents=True,exist_ok=True,mode=0o700)
    if target.is_symlink() or target.stat().st_mode & 0o022:
        fail("prefix permissions/identity unsafe")
    final = target/rid
    if final.exists() or final.is_symlink():
        info = verify(final,expected=files)
        return dict(info,path=str(final),result="already_verified")
    staging = Path(tempfile.mkdtemp(prefix=".staging-",dir=target))
    try:
        os.chmod(staging,0o700)
        for name in FILES:
            data = _checked_file(Path(source_dir)/name)
            if digest(data) != files[name]:
                fail("source changed during installation")
            output=staging/name
            with output.open("xb") as fp:
                fp.write(data)
                fp.flush()
                os.fsync(fp.fileno())
            os.chmod(output,0o755 if name.endswith(".py") else 0o644)
        manifest = {"schema":SCHEMA,"release_id":rid,"files":files}
        serialized=(json.dumps(manifest,sort_keys=True,separators=(",",":"))+"\n").encode()
        with (staging/MANIFEST).open("xb") as fp:
            fp.write(serialized)
            fp.flush()
            os.fsync(fp.fileno())
        os.chmod(staging/MANIFEST,0o644)
        verify(staging,expected=files)
        # Atomically publish to a unique content-addressed path.
        if final.exists() or final.is_symlink():
            fail("destination concurrently appeared, refusing overwrite")
        staging.rename(final)
        fd=os.open(target,os.O_RDONLY|getattr(os,"O_DIRECTORY",0))
        try: os.fsync(fd)
        finally: os.close(fd)
        return dict(verify(final,expected=files),path=str(final),result="installed")
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prefix",type=Path,
        default=_account_home()/".exp053-readonly-preflight")
    p.add_argument("--verify",type=Path,default=None)
    p.add_argument("--install",action="store_true")
    args=p.parse_args(argv)
    if args.verify:
        result=verify(args.verify)
    elif args.install:
        result=install(args.prefix)
    else:
        files=source_inventory()
        result={"schema":SCHEMA,"release_id":release_id(files),
                "files":files,"result":"dry_run_no_files_written"}
    print(json.dumps(result,sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
