#!/usr/bin/env python3
"""Source-free, read-only EXP-053 human-study owner readiness check.

Only metadata of private assets is inspected; licensed RACE text, reviewer
forms, reviewer identities and signatures are never read. Always NO-GO.
"""

import hashlib
import json
import os
import stat
from pathlib import Path

import exp053_race_review as review

SCHEMA = "exp053-review-owner-readiness-v1"
SPEC = Path(__file__).resolve().parent / "EXP053-REVIEW-OWNER-READINESS-SPEC-2026-10-19.md"
SPEC_SHA = "f07806c897e035f6c9c73413c4334c14a98e488d4acac68fb548c1cadbee46a8"
PRIVATE_DIR = Path("/media/scott/SSD_4TB/exp053-private-20261016")
PRIVATE_FILES = ("packet.json", "sealed_exam_key.json", "reviewer_A.json", "reviewer_B.json")
REQUIRED_KEYS = {"schema", "source", "source_sha256", "protocol_sha256", "packet_sha256",
                 "sealed_answer_key_sha256", "candidate_articles", "human_reviewed_articles",
                 "minimum_human_reviewed", "evaluation_admitted", "copyright", "rows"}
ROW_KEYS = {"case_id", "article_sha256", "question_sha256", "options_sha256", "article_chars"}
MAX_MANIFEST = 131072
MAX_PROTOCOL = 65536


class Hold(ValueError):
    """Sanitized refusal. Never expose input text or private paths."""


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _hex(value, n=64):
    return isinstance(value, str) and len(value) == n and all(c in "0123456789abcdef" for c in value)


def _public_bytes(path, max_size):
    try:
        path = Path(path)
        if path.is_symlink():
            raise Hold("public reviewed artifact unavailable")
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | os.O_NONBLOCK)
        with os.fdopen(fd, "rb") as inp:
            info = os.fstat(inp.fileno())
            if not stat.S_ISREG(info.st_mode) or not 1 <= info.st_size <= max_size:
                raise Hold("public reviewed artifact invalid")
            data = inp.read(max_size + 1)
            after = os.fstat(inp.fileno())
            if (len(data) != info.st_size or info.st_dev != after.st_dev or
                    info.st_ino != after.st_ino or info.st_mtime_ns != after.st_mtime_ns):
                raise Hold("public reviewed artifact changed")
        return data
    except OSError as exc:
        raise Hold("public reviewed artifact unavailable") from exc


def _no_duplicates(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise Hold("ambiguous public manifest")
        result[name] = value
    return result


def _invalid_constant(_):
    raise Hold("non-finite public manifest number")


def _validate_public(manifest):
    if not isinstance(manifest, dict) or set(manifest) != REQUIRED_KEYS:
        raise Hold("public manifest schema mismatch")
    if (manifest["schema"] != "exp053-public-manifest-v1" or
            manifest["source_sha256"] != review.SOURCE_SHA or
            manifest["protocol_sha256"] != review.PROTOCOL_SHA or
            manifest["candidate_articles"] != 64 or
            manifest["minimum_human_reviewed"] != 40 or
            type(manifest["human_reviewed_articles"]) is not int or
            manifest["human_reviewed_articles"] != 0 or
            manifest["evaluation_admitted"] is not False or
            not all(_hex(manifest[k]) for k in
                    ("packet_sha256", "sealed_answer_key_sha256"))):
        raise Hold("public cohort authority unrecognized or drifted")
    rows = manifest["rows"]
    if not isinstance(rows, list) or len(rows) != 64:
        raise Hold("public cohort rows unavailable")
    cases = set()
    articles = set()
    for row in rows:
        if (not isinstance(row, dict) or set(row) != ROW_KEYS or
                not _hex(row["case_id"], 20) or
                not all(_hex(row[k]) for k in
                        ("article_sha256", "question_sha256", "options_sha256")) or
                type(row["article_chars"]) is not int or not 250 <= row["article_chars"] <= 3500):
            raise Hold("public cohort row not source-free or malformed")
        if row["case_id"] in cases or row["article_sha256"] in articles:
            raise Hold("public cohort repeats article or case identity")
        cases.add(row["case_id"])
        articles.add(row["article_sha256"])
    return True


def _private_status(folder, names):
    """Report private filenames using directory-relative METADATA only.

    Open *only the containing directory*, never the private data files. A
    switched parent path or symlink cannot be mistaken for verified current
    contents. Anything uncertain fails closed to BLOCKED_UNVERIFIED.
    """
    blocked = {name: "BLOCKED_UNVERIFIED" for name in names}
    if not getattr(os, "O_DIRECTORY", 0) or not getattr(os, "O_NOFOLLOW", 0):
        return blocked
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    try:
        parent_fd = os.open(folder, flags)
    except OSError:
        return blocked
    try:
        d = os.fstat(parent_fd)
        if (not stat.S_ISDIR(d.st_mode) or d.st_uid != os.geteuid() or
                d.st_mode & 0o077):
            return blocked
        found = dict(blocked)
        for name in names:
            try:
                info = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
                if (stat.S_ISREG(info.st_mode) and info.st_uid == os.geteuid() and
                        info.st_size > 0 and info.st_mode & 0o077 == 0):
                    found[name] = "PRESENT_METADATA_ONLY_UNVERIFIED"
            except OSError:
                pass
        try:
            refreshed_fd = os.open(folder, flags)
        except OSError:
            return blocked
        try:
            refreshed = os.fstat(refreshed_fd)
            current = os.fstat(parent_fd)
            if ((refreshed.st_dev, refreshed.st_ino) != (d.st_dev, d.st_ino) or
                    (current.st_dev, current.st_ino) != (d.st_dev, d.st_ino)):
                return blocked
            return found
        finally:
            os.close(refreshed_fd)
    except OSError:
        return blocked
    finally:
        os.close(parent_fd)


def inspect(*, manifest_path=review.MANIFEST, protocol_path=review.PROTOCOL,
            private_dir=PRIVATE_DIR, source_path=review.SOURCE,
            pinned_manifest_sha=review.MANIFEST_SHA,
            pinned_protocol_sha=review.PROTOCOL_SHA,
            reviewer_pin=review.APPROVED_REVIEWER_ROSTER_SHA256,
            spec_path=SPEC, pinned_spec_sha=SPEC_SHA):
    if _sha(_public_bytes(spec_path, MAX_PROTOCOL)) != pinned_spec_sha:
        raise Hold("research-only preflight specification drift")
    if _sha(_public_bytes(protocol_path, MAX_PROTOCOL)) != pinned_protocol_sha:
        raise Hold("frozen preregistered protocol differs")
    raw = _public_bytes(manifest_path, MAX_MANIFEST)
    if not _hex(pinned_manifest_sha) or _sha(raw) != pinned_manifest_sha:
        raise Hold("public source-free cohort manifest differs")
    try:
        manifest = json.loads(raw, object_pairs_hook=_no_duplicates,
                              parse_constant=_invalid_constant)
    except (ValueError, TypeError, UnicodeError, RecursionError) as exc:
        raise Hold("public source-free manifest invalid") from exc
    _validate_public(manifest)
    private = _private_status(private_dir, PRIVATE_FILES)
    try:
        source_stat = os.lstat(source_path)
        if stat.S_ISLNK(source_stat.st_mode):
            source_status = "SYMLINK_PRESENT_TARGET_UNVERIFIED"
        elif stat.S_ISREG(source_stat.st_mode) and source_stat.st_size > 0:
            source_status = "PRESENT_METADATA_ONLY_UNVERIFIED"
        else:
            source_status = "BLOCKED_UNVERIFIED"
    except OSError:
        source_status = "BLOCKED_UNVERIFIED"
    return {
        "schema": SCHEMA,
        "decision": "NO_GO_RESEARCH_ONLY",
        "scientific_execution_authorized": False,
        "independent_human_review_authorized": False,
        "reviewer_enrollment_status": "UNAPPROVED_OWNER_PIN" if not _hex(reviewer_pin) else "PIN_CONFIGURED_NOT_IDENTITY_PROOF",
        "admitted_human_review_count": 0 if not _hex(reviewer_pin) else "UNKNOWN_REQUIRES_PRIVATE_ADMISSION_GATE",
        "frozen_manifest_human_reviewed_articles": 0,
        "candidate_article_count": 64,
        "required_distinct_co_reviewed_articles": 40,
        "minimum_two_reviewer_case_decisions": 80,
        "blind_review_slots": ["A", "B"],
        "requires_independent_adjudicator_for_disputes": True,
        "requires_independent_human_identity_and_key_custody": True,
        "copyright_restriction": "NONCOMMERCIAL_RESEARCH_NO_SOURCE_REDISTRIBUTION",
        "corpus": source_status,
        "private_assets_metadata_only": private,
        "private_content_accessed": False,
        "next_action": "HUMAN_OWNER_VALIDATE_TWO_INDEPENDENT_REVIEWERS_AND_PRIVATE_KEY_CUSTODY",
    }


def main():
    try:
        print(json.dumps(inspect(), sort_keys=True, separators=(",", ":")))
    except (Hold, OSError, ValueError):
        print('{"decision":"NO_GO_RESEARCH_ONLY","scientific_execution_authorized":false,'
              '"independent_human_review_authorized":false,'
              '"status":"PUBLIC_SOURCE_FREE_MANIFEST_INVALID"}')
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
