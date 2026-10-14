"""Manufactured, data-free checks for coordinated marked LaTeX updates."""

import concurrent.futures
import importlib.util
import json
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import threading
import time
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "update_paper_section.py"
SPEC = importlib.util.spec_from_file_location("update_paper_section", SCRIPT)
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)


def marked(section, content):
    return f"% BEGIN GFP SECTION {section}\n{content}% END GFP SECTION {section}\n"


class CoordinatedPaperTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="gfp-paper-tool-test-", dir="/private/tmp")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.paper = self.root / "manufactured.tex"
        self.original = (
            "\\documentclass{article}\n\\begin{document}\n"
            + marked("contactgen", "Old ContactGen.\n")
            + marked("unidexgrasp", "Old UniDexGrasp.\n")
            + marked("dexgrasp-anything", "Old DexGrasp Anything.\n")
            + "\\end{document}\n"
        ).encode()
        self.paper.write_bytes(self.original)

    def write(self, section, content, expected=None):
        if expected is None:
            expected = TOOL.inspect_section(self.paper, section)["section_sha256"]
        return TOOL.update_section(self.paper, section, content, expected)

    def test_simultaneous_distinct_blocks_merge_without_lost_updates(self):
        # Real subprocesses exercise POSIX locks and atomic replacement, not
        # merely threads that happen to serialize within one Python process.
        for round_number in range(4):
            sections = ["contactgen", "unidexgrasp", "dexgrasp-anything"]
            snapshots = {s: TOOL.inspect_section(self.paper, s) for s in sections}
            gate = threading.Barrier(len(sections))

            def perform(section):
                fragment = self.root / f"{section}-{round_number}.tex"
                text = f"New {section}, round {round_number}; μm.\n"
                fragment.write_text(text, encoding="utf-8")
                gate.wait(timeout=10)
                completed = subprocess.run(
                    [sys.executable, str(SCRIPT), "--paper", str(self.paper), "--section", section,
                     "--fragment-file", str(fragment), "--expected-section-sha256",
                     snapshots[section]["section_sha256"]],
                    capture_output=True, text=True, timeout=20,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
                return section, text

            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                outcomes = list(executor.map(perform, sections))
            for section, text in outcomes:
                self.assertEqual(TOOL.inspect_section(self.paper, section)["content"], text)
            data = self.paper.read_bytes()
            self.assertTrue(data.startswith(b"\\documentclass{article}\n\\begin{document}\n"))
            self.assertTrue(data.endswith(b"\\end{document}\n"))

    def test_same_block_stale_snapshot_refused_without_overwrite(self):
        expected = TOOL.inspect_section(self.paper, "contactgen")["section_sha256"]
        self.write("contactgen", b"First author's update.\n", expected)
        before = self.paper.read_bytes()
        with self.assertRaisesRegex(TOOL.PaperSectionError, "Stale"):
            self.write("contactgen", b"Second author's stale update.\n", expected)
        self.assertEqual(self.paper.read_bytes(), before)

    def test_reread_under_lock_preserves_unrelated_intervening_update(self):
        expected = TOOL.inspect_section(self.paper, "contactgen")["section_sha256"]
        self.write("unidexgrasp", b"An unrelated intervening update.\n")
        self.write("contactgen", b"Current ContactGen update.\n", expected)
        self.assertEqual(TOOL.inspect_section(self.paper, "unidexgrasp")["content"],
                         "An unrelated intervening update.\n")

    def test_writer_waits_for_lock_and_rereads_after_release(self):
        expected = TOOL.inspect_section(self.paper, "contactgen")["section_sha256"]
        fragment = self.root / "delayed-fragment.tex"
        fragment.write_bytes(b"Writer update after release.\n")
        canonical = TOOL.canonical_paper_path(self.paper)
        with TOOL.paper_lock(canonical):
            process = subprocess.Popen(
                [sys.executable, str(SCRIPT), "--paper", str(self.paper), "--section", "contactgen",
                 "--fragment-file", str(fragment), "--expected-section-sha256", expected],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )
            self.addCleanup(lambda: process.kill() if process.poll() is None else None)
            time.sleep(0.2)
            self.assertIsNone(process.poll(), "Writer should be blocked by the advisory lock")
            # This is an intervening cooperating update while the test owns the
            # same lock. A writer that took its snapshot before acquiring it
            # would erase this update after the lock was released.
            data = self.paper.read_bytes().replace(
                b"Old UniDexGrasp.\n", b"Protected intervening update.\n"
            )
            self.paper.write_bytes(data)
        stdout, stderr = process.communicate(timeout=10)
        self.assertEqual(process.returncode, 0, stderr)
        self.assertEqual(json.loads(stdout)["status"], "updated")
        self.assertEqual(TOOL.inspect_section(self.paper, "unidexgrasp")["content"],
                         "Protected intervening update.\n")

    def test_duplicate_missing_reversed_nested_malformed_markers_refused(self):
        examples = [
            marked("contactgen", "One.\n") + marked("contactgen", "Duplicate.\n"),
            "% BEGIN GFP SECTION contactgen\nMissing end.\n",
            "% END GFP SECTION contactgen\n% BEGIN GFP SECTION contactgen\n",
            "% BEGIN GFP SECTION contactgen\n% BEGIN GFP SECTION other\n"
            "% END GFP SECTION other\n% END GFP SECTION contactgen\n",
            "% BEGIN GFP SECTION contactgen\n% END GFP SECTION other\n",
            "% BEGIN GFP SECTION contactgen extra\nBody.\n% END GFP SECTION contactgen\n",
            "% BEGIN GFP SECTION contactgen",
            "Inline % BEGIN GFP SECTION contactgen\n",
        ]
        for source in examples:
            with self.subTest(source=source):
                self.paper.write_text(source)
                before = self.paper.read_bytes()
                with self.assertRaises(TOOL.PaperSectionError):
                    self.write("contactgen", b"Attempted replacement.\n", "0" * 64)
                self.assertEqual(self.paper.read_bytes(), before)

    def test_fragment_reserved_markers_and_document_boundaries_refused(self):
        examples = [
            b"% BEGIN GFP SECTION sneaky\n", b"% END GFP SECTION sneaky\n",
            b"Text % BEGIN GFP SECTION sneaky\n", b"\\begin{document}\n",
            b"\\end{document}\n", b"\\begin { document }\n", b"No terminal newline",
            b"Invalid UTF-8: \xff\n",
        ]
        for fragment in examples:
            with self.subTest(fragment=fragment):
                with self.assertRaises((TOOL.PaperSectionError, UnicodeError)):
                    self.write("contactgen", fragment)
                self.assertEqual(self.paper.read_bytes(), self.original)

    def test_unchanged_keeps_bytes_inode_and_permissions(self):
        self.paper.chmod(0o640)
        info = self.paper.stat()
        result = self.write("contactgen", b"Old ContactGen.\n")
        self.assertEqual(result["status"], "unchanged")
        self.assertEqual(self.paper.read_bytes(), self.original)
        after = self.paper.stat()
        self.assertEqual(after.st_ino, info.st_ino)
        self.assertEqual(after.st_mtime_ns, info.st_mtime_ns)
        self.assertEqual(stat.S_IMODE(after.st_mode), 0o640)

    def test_replacement_preserves_permissions_and_other_bytes(self):
        self.paper.chmod(0o640)
        result = self.write("contactgen", "New UTF-8 α.\n".encode())
        expected = self.original.replace(b"Old ContactGen.\n", "New UTF-8 α.\n".encode())
        self.assertEqual(result["status"], "updated")
        self.assertEqual(self.paper.read_bytes(), expected)
        self.assertEqual(stat.S_IMODE(self.paper.stat().st_mode), 0o640)
        self.assertEqual(result["full_file_sha256"], TOOL.sha256(expected))
        self.assertEqual(list(self.root.glob(".manufactured.tex.section-*")), [])

    def test_missing_section_and_invalid_expected_hash_refused(self):
        for section, expected in [("absent", "0" * 64), ("contactgen", "invalid"),
                                  ("bad section", "0" * 64)]:
            with self.subTest(section=section, expected=expected):
                with self.assertRaises(TOOL.PaperSectionError):
                    self.write(section, b"Attempted replacement.\n", expected)
                self.assertEqual(self.paper.read_bytes(), self.original)

    def test_empty_fragment_and_crlf_paper_remain_well_formed(self):
        self.paper.write_bytes(self.original.replace(b"\n", b"\r\n"))
        self.write("contactgen", b"")
        inspected = TOOL.inspect_section(self.paper, "contactgen")
        self.assertEqual(inspected["content"], "")
        self.assertEqual(inspected["section_sha256"], TOOL.sha256(b""))
        self.assertIn(b"% END GFP SECTION contactgen\r\n", self.paper.read_bytes())

    def test_canonical_aliases_share_lock(self):
        alias = self.root / "alias.tex"
        alias.symlink_to(self.paper)
        canonical = TOOL.canonical_paper_path(self.paper)
        self.assertEqual(TOOL.lock_path(canonical), TOOL.lock_path(TOOL.canonical_paper_path(alias)))
        self.write("contactgen", b"Canonical update.\n")
        self.assertEqual(alias.read_bytes(), self.paper.read_bytes())

    def test_cli_inspect_and_required_write_arguments(self):
        inspected = subprocess.run(
            [sys.executable, str(SCRIPT), "--paper", str(self.paper), "--section", "contactgen", "--inspect"],
            capture_output=True, text=True, timeout=10,
        )
        self.assertEqual(inspected.returncode, 0, inspected.stderr)
        result = json.loads(inspected.stdout)
        self.assertEqual(result["content"], "Old ContactGen.\n")
        self.assertEqual(result["section_sha256"], TOOL.sha256(b"Old ContactGen.\n"))
        self.assertEqual(result["full_file_sha256"], TOOL.sha256(self.original))
        refused = subprocess.run(
            [sys.executable, str(SCRIPT), "--paper", str(self.paper), "--section", "contactgen"],
            capture_output=True, text=True, timeout=10,
        )
        self.assertEqual(refused.returncode, 2)
        self.assertEqual(self.paper.read_bytes(), self.original)


if __name__ == "__main__":
    unittest.main()
