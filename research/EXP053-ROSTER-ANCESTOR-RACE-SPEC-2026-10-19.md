# EXP-053 owner-roster parent-directory substitution defense

**2026-10-19, frozen before implementation. Research-only: NO_GO_RESEARCH_ONLY.**

Reproduced with disposable synthetic roster files: between
`path.resolve(strict=True)` and `os.open(path, O_NOFOLLOW)` an adversary may
rename the roster's immediate parent directory and replace that path with a
symlink into the public repository. Although the **final file** is regular,
`O_NOFOLLOW` alone does not protect *ancestor* components. When the substitute
file is byte-identical to the pinned roster, the loader previously accepted a
roster read from the public repository. This violates explicit repo exclusion
and is a source-provenance/custody-path race, not a signature forgery.

Narrow source-only remedy:

1. Resolve the intended user path to an absolute lexical pathname (not
   following symlinks). Open from `/` and traverse **every directory
   component** using directory-file-descriptor-relative
   `O_PATH|O_DIRECTORY|O_NOFOLLOW`, closing intermediate fd handles. Open
   the final roster with `O_RDONLY|O_NOFOLLOW|O_NONBLOCK`, relative to the
   verified immediate-parent fd. Require these OS flags or refuse. Linux
   `O_PATH` supports directory traversal without a read grant on `/`, which
   the research host's restricted fleet environment otherwise rejects.
2. Verify a regular single-link mode-0600 file owned by the effective user,
   size 1..16384, then read its bounded bytes with `os.read` using the one
   file fd. Check pre/post `fstat()` device/inode/size/mtime/ctime. Protect
   all raw fds against exceptions. No roster bytes or paths in errors.
3. Ensure the roster's absolute resolved path is not inside the public repo,
   both before and after reading. After reading, repeat the secure full-path
   traversal and compare new and original file `(st_dev, st_ino)`; refuse if
   a directory was substituted, renamed, moved or lost during inspection.
4. Tests simulate swapping the immediate and ancestor directory to a symlink
   into the repo, a file-swap during the read, symlinked ancestors, and
   normal owner-private non-symlinked control. Retain SHA pin, packet binding,
   blind-slot and key uniqueness tests. All material synthetic; no access to
   actual RACE text or private reviewer files, no model invocation.
5. This check does not defeat a privileged/same-UID attacker who can replace
   inodes after the final check, and cannot supply independence of witnesses.
   It does not enroll reviewers or change release authority.
