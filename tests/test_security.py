"""Tests for the safety rules applied to untrusted evidence files.

Each test drives a real attack shape through the real code path, so these fail
loudly if a guard is ever weakened.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from ingestion.errors import ResourceLimitError, SecurityError
from ingestion.models import MediaType
from ingestion.scanner import CaseScanner
from ingestion.security import (
    assert_regular_file,
    assert_safe_external_path,
    assert_within_size_limit,
    classify,
    resolve_within,
    sha256_file,
)


class TestPathContainment:
    def test_symlink_escaping_the_case_folder_is_rejected(self, case_folder, tmp_path):
        outside = tmp_path / "secrets.txt"
        outside.write_text("private")
        link = case_folder / "evidence.txt"
        link.symlink_to(outside)

        with pytest.raises(SecurityError, match="escapes the case folder"):
            resolve_within(case_folder, link)

    def test_parent_traversal_is_rejected(self, case_folder):
        with pytest.raises(SecurityError, match="escapes the case folder"):
            resolve_within(case_folder, Path("../../etc/passwd"))

    def test_ordinary_nested_file_is_allowed(self, case_folder):
        nested = case_folder / "video" / "clip.mp4"
        nested.parent.mkdir()
        nested.write_bytes(b"data")
        assert resolve_within(case_folder, nested) == nested.resolve()

    def test_scanner_skips_symlinked_evidence(self, config, case_folder, tmp_path):
        outside = tmp_path / "outside.pdf"
        outside.write_bytes(b"%PDF-1.4\n")
        (case_folder / "linked.pdf").symlink_to(outside)

        result = CaseScanner(config).scan()

        assert result.files == []
        assert len(result.skipped) == 1
        assert "escapes the case folder" in result.skipped[0].detail


class TestFileTypeGuards:
    def test_non_regular_files_are_rejected(self, case_folder):
        fifo = case_folder / "pipe"
        os.mkfifo(fifo)
        with pytest.raises(SecurityError, match="not a regular file"):
            assert_regular_file(fifo)

    def test_content_wins_over_a_spoofed_extension(self, case_folder):
        """A PDF renamed to .mp4 must not reach the video decoder."""
        spoofed = case_folder / "evidence.mp4"
        spoofed.write_bytes(b"%PDF-1.4\n%\xc7\xec\x8f\xa2\ntrailer\n")

        media_type, mime, mismatch = classify(spoofed, declared=MediaType.VIDEO)

        assert media_type is MediaType.PDF
        assert mime == "application/pdf"
        assert mismatch is True

    def test_matching_declaration_reports_no_mismatch(self, case_folder):
        real_pdf = case_folder / "report.pdf"
        real_pdf.write_bytes(b"%PDF-1.4\n%\xc7\xec\x8f\xa2\ntrailer\n")

        media_type, _, mismatch = classify(real_pdf, declared=MediaType.PDF)

        assert media_type is MediaType.PDF
        assert mismatch is False

    def test_extension_masquerade_is_caught_without_a_declaration(self, case_folder):
        """An undeclared file has no config claim to contradict, so the
        extension is what the content is checked against."""
        spoofed = case_folder / "evidence.mp4"
        spoofed.write_bytes(b"%PDF-1.4\n%\xc7\xec\x8f\xa2\ntrailer\n")

        media_type, _, mismatch = classify(spoofed, declared=None)

        assert media_type is MediaType.PDF
        assert mismatch is True

    def test_mismatch_is_recorded_on_the_scanned_file(self, config, case_folder):
        spoofed = case_folder / "clip.mp4"
        spoofed.write_bytes(b"%PDF-1.4\n%\xc7\xec\x8f\xa2\ntrailer\n")

        scanned = CaseScanner(config).scan().files[0]

        assert scanned.media_type is MediaType.PDF
        assert scanned.type_mismatch is True
        assert scanned.detected_mime == "application/pdf"


class TestResourceLimits:
    def test_oversized_file_is_rejected(self, case_folder):
        big = case_folder / "big.bin"
        big.write_bytes(b"x" * 1024)
        with pytest.raises(ResourceLimitError, match="over the"):
            assert_within_size_limit(big, max_bytes=512)

    def test_scanner_skips_oversized_files(self, config, case_folder):
        config.limits.max_file_bytes = 8
        oversized = case_folder / "big.pdf"
        oversized.write_bytes(b"%PDF-1.4\n" + b"x" * 100)

        result = CaseScanner(config).scan()

        assert result.files == []
        assert "over the" in result.skipped[0].detail


class TestExternalCommandSafety:
    def test_relative_path_is_refused(self):
        """Absolute paths cannot be mistaken for a CLI flag; relative ones can."""
        with pytest.raises(SecurityError, match="absolute path"):
            assert_safe_external_path(Path("-i"))

    def test_absolute_path_is_allowed(self, tmp_path):
        target = tmp_path / "clip.wav"
        assert assert_safe_external_path(target) == str(target)


class TestHashing:
    def test_sha256_matches_known_vector(self, case_folder):
        target = case_folder / "abc.txt"
        target.write_bytes(b"abc")
        assert sha256_file(target) == (
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        )
