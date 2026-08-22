"""Discovery of evidence files on disk.

The scanner's only job is to turn a folder into a validated, hashed list of
ScannedFile records. It does not talk to the database and does not extract
anything, so it can be pointed at a fixture directory in a test.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from .config import AppConfig
from .errors import IngestionError
from .models import FileReport, MediaType, ScannedFile
from .security import (
    assert_regular_file,
    assert_within_size_limit,
    classify,
    resolve_within,
    sha256_file,
)

log = logging.getLogger(__name__)


@dataclass
class ScanResult:
    files: list[ScannedFile] = field(default_factory=list)
    skipped: list[FileReport] = field(default_factory=list)


class CaseScanner:
    """Walks the case folder and produces validated, hashed file records."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config

    def scan(self) -> ScanResult:
        case_folder = self._config.case_folder
        if not case_folder.is_dir():
            raise IngestionError(f"case folder not found: {case_folder}")

        result = ScanResult()
        for path in self._walk(case_folder):
            try:
                scanned = self._inspect(path, case_folder)
            except IngestionError as exc:
                log.warning("skipping %s: %s", path.name, exc)
                result.skipped.append(
                    FileReport(
                        file_name=path.name,
                        media_type="unknown",
                        status="skipped",
                        detail=str(exc),
                    )
                )
                continue

            if scanned is None:
                continue
            result.files.append(scanned)

        log.info(
            "scanned %s: %d file(s) accepted, %d skipped",
            case_folder, len(result.files), len(result.skipped),
        )
        return result

    def _walk(self, case_folder: Path) -> list[Path]:
        """List candidate files in deterministic order.

        `recurse_symlinks=False` keeps a symlinked directory from pulling the
        walk outside the case folder — or into an infinite loop.
        """
        candidates = [
            path
            for path in case_folder.rglob("*", recurse_symlinks=False)
            if not path.is_dir() and not self._is_hidden(path, case_folder)
        ]
        return sorted(candidates)

    @staticmethod
    def _is_hidden(path: Path, case_folder: Path) -> bool:
        """Ignore dotfiles such as .DS_Store and editor swap files."""
        relative = path.relative_to(case_folder)
        return any(part.startswith(".") for part in relative.parts)

    def _inspect(self, path: Path, case_folder: Path) -> ScannedFile | None:
        """Validate, classify, and hash one candidate file."""
        resolved = resolve_within(case_folder, path)
        assert_regular_file(resolved)
        size_bytes = assert_within_size_limit(resolved, self._config.limits.max_file_bytes)

        relative_path = resolved.relative_to(case_folder).as_posix()
        entry = self._config.declared_entry(relative_path)
        declared: MediaType | None = entry.type if entry else None

        media_type, detected_mime, mismatch = classify(resolved, declared)
        if media_type is None:
            log.debug("ignoring %s: unrecognised media type", relative_path)
            return None

        # Hashing before any processing fixes the evidence's identity at the
        # moment of ingestion, which is what the chain of custody rests on.
        digest = sha256_file(resolved)

        return ScannedFile(
            path=resolved,
            file_name=resolved.name,
            media_type=media_type,
            sha256=digest,
            size_bytes=size_bytes,
            declared_type=declared.value if declared else None,
            detected_mime=detected_mime,
            type_mismatch=mismatch,
            author=entry.author if entry else None,
            created_date=entry.created_date if entry else None,
            metadata=self._entry_metadata(entry, relative_path),
        )

    @staticmethod
    def _entry_metadata(entry, relative_path: str) -> dict:
        """Carry the investigator's free-form config metadata onto the record."""
        base = {"relative_path": relative_path}
        if entry is None:
            base["declared_in_config"] = False
            return base

        extras = entry.model_dump(exclude_none=True)
        extras.pop("path", None)
        extras.pop("type", None)
        if isinstance(extras.get("created_date"), object) and entry.created_date:
            extras["created_date"] = entry.created_date.isoformat()
        base.update(extras)
        base["declared_in_config"] = True
        return base
