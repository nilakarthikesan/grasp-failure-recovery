#!/usr/bin/env python3
"""Merge one marked UTF-8 LaTeX section without overwriting other sections.

Inspect a section before editing and supply its reported section SHA-256 when
writing. Each cooperating process locks the same canonical paper path, rereads
the paper under that lock, and replaces only the selected section. Fragments are
inserted verbatim and must be empty or end in a newline.

Locks are advisory: editors, scripts and agents that bypass this tool can still
overwrite it. Save editor changes before inspecting; do not save a stale editor
buffer over coordinated updates. Atomic replacement preserves permission bits.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
import tempfile
from typing import Iterator


SECTION_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
MARKER_LINE = re.compile(
    r"[ \t]*%[ \t]+(BEGIN|END)[ \t]+GFP[ \t]+SECTION[ \t]+"
    r"([A-Za-z0-9][A-Za-z0-9_.-]*)[ \t]*(?:\r?\n)?\Z"
)
RESERVED_MARKER = re.compile(r"%[ \t]*(?:BEGIN|END)[ \t]+GFP[ \t]+SECTION\b")
DOCUMENT_BOUNDARY = re.compile(r"\\(?:begin|end)\s*\{\s*document\s*\}")


class PaperSectionError(ValueError):
    """A refused operation; the paper has not been changed."""


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_paper_path(path: str | Path) -> Path:
    result = Path(path).expanduser().resolve(strict=True)
    if not result.is_file():
        raise PaperSectionError(f"Paper is not a regular file: {result}")
    return result


def lock_path(paper: Path) -> Path:
    """All users of this tool share this per-user, canonical-path lock."""
    directory = Path("/private/tmp") / f"gfp-paper-section-locks-{os.getuid()}"
    try:
        directory.mkdir(mode=0o700)
    except FileExistsError:
        pass
    info = directory.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid():
        raise PaperSectionError("Paper lock directory must be owned by this user")
    if stat.S_IMODE(info.st_mode) & 0o077:
        raise PaperSectionError("Paper lock directory must have private permissions")
    return directory / (sha256(os.fsencode(paper)) + ".lock")


@contextmanager
def paper_lock(paper: Path) -> Iterator[None]:
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(lock_path(paper), flags, 0o600)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
            raise PaperSectionError("Paper lock must be a regular file owned by this user")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        os.close(descriptor)


def section_ranges(data: bytes) -> dict[str, tuple[int, int]]:
    """Validate every marker pair and return the exact content byte ranges."""
    text = data.decode("utf-8")
    ranges: dict[str, tuple[int, int]] = {}
    seen: set[str] = set()
    opened: tuple[str, int] | None = None
    offset = 0
    for line in text.splitlines(keepends=True):
        length = len(line.encode("utf-8"))
        match = MARKER_LINE.fullmatch(line)
        if not match:
            if RESERVED_MARKER.search(line):
                raise PaperSectionError("Malformed or inline reserved section marker")
            offset += length
            continue
        kind, section = match.groups()
        if kind == "BEGIN":
            if section in seen:
                raise PaperSectionError(f"Duplicate section marker: {section}")
            if opened is not None:
                raise PaperSectionError("Nested or missing END section marker")
            if not line.endswith("\n"):
                raise PaperSectionError("BEGIN marker must end with a newline")
            seen.add(section)
            opened = (section, offset + length)
        else:
            if opened is None or opened[0] != section:
                raise PaperSectionError(f"Missing, reversed or mismatched markers: {section}")
            ranges[section] = (opened[1], offset)
            opened = None
        offset += length
    if opened is not None:
        raise PaperSectionError(f"Missing END marker: {opened[0]}")
    return ranges


def section_bytes(data: bytes, section: str) -> tuple[bytes, tuple[int, int]]:
    if not SECTION_ID.fullmatch(section):
        raise PaperSectionError("Section ID must contain only letters, digits, '.', '_' or '-'")
    ranges = section_ranges(data)
    if section not in ranges:
        raise PaperSectionError(f"Missing section markers: {section}")
    start, end = ranges[section]
    return data[start:end], (start, end)


def validate_fragment(fragment: bytes) -> None:
    text = fragment.decode("utf-8")
    if RESERVED_MARKER.search(text):
        raise PaperSectionError("Fragment contains a reserved section marker")
    if DOCUMENT_BOUNDARY.search(text):
        raise PaperSectionError("Fragment contains a begin/end document boundary")
    if fragment and not fragment.endswith(b"\n"):
        raise PaperSectionError("A nonempty fragment must end with a newline")


def atomic_replace(paper: Path, data: bytes, original_mode: int) -> None:
    descriptor, name = tempfile.mkstemp(prefix=f".{paper.name}.section-", dir=paper.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fchmod(handle.fileno(), original_mode)
            os.fsync(handle.fileno())
        os.replace(temporary, paper)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def inspect_section(paper: str | Path, section: str) -> dict[str, object]:
    canonical = canonical_paper_path(paper)
    with paper_lock(canonical):
        data = canonical.read_bytes()
        content, _ = section_bytes(data, section)
        return {
            "status": "inspected",
            "paper": str(canonical),
            "section": section,
            "section_sha256": sha256(content),
            "full_file_sha256": sha256(data),
            "content": content.decode("utf-8"),
        }


def update_section(
    paper: str | Path, section: str, fragment: bytes, expected_section_sha256: str
) -> dict[str, object]:
    if not re.fullmatch(r"[0-9a-fA-F]{64}", expected_section_sha256):
        raise PaperSectionError("Expected section SHA-256 must be 64 hexadecimal characters")
    validate_fragment(fragment)
    canonical = canonical_paper_path(paper)
    with paper_lock(canonical):
        info = canonical.stat()
        if not stat.S_ISREG(info.st_mode):
            raise PaperSectionError("Paper is no longer a regular file")
        data = canonical.read_bytes()
        content, (start, end) = section_bytes(data, section)
        if sha256(content) != expected_section_sha256.lower():
            raise PaperSectionError(
                f"Stale section {section}: inspect again and reconcile its current content"
            )
        updated = data[:start] + fragment + data[end:]
        # Check the complete result before replacing the paper, including all
        # unrelated markers; neither fragments nor this merge may alter them.
        section_bytes(updated, section)
        changed = updated != data
        if changed:
            atomic_replace(canonical, updated, stat.S_IMODE(info.st_mode))
        return {
            "status": "updated" if changed else "unchanged",
            "paper": str(canonical),
            "section": section,
            "previous_section_sha256": sha256(content),
            "section_sha256": sha256(fragment),
            "previous_full_file_sha256": sha256(data),
            "full_file_sha256": sha256(updated),
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper", required=True, type=Path)
    parser.add_argument("--section", required=True)
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--fragment-file", type=Path)
    parser.add_argument("--expected-section-sha256")
    args = parser.parse_args(argv)
    if args.inspect:
        if args.fragment_file is not None or args.expected_section_sha256 is not None:
            parser.error("--inspect cannot be combined with write arguments")
    elif args.fragment_file is None or args.expected_section_sha256 is None:
        parser.error("Writing requires --fragment-file and --expected-section-sha256")
    try:
        if args.inspect:
            result = inspect_section(args.paper, args.section)
        else:
            result = update_section(
                args.paper,
                args.section,
                args.fragment_file.read_bytes(),
                args.expected_section_sha256,
            )
    except (PaperSectionError, OSError, UnicodeError) as error:
        print(json.dumps({"status": "refused", "error": str(error)}), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
