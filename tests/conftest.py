"""Shared fixtures. Nothing here touches Postgres — these tests are offline."""
from __future__ import annotations

from pathlib import Path

import pytest

from ingestion.config import AppConfig, CaseInfo, Paths


@pytest.fixture
def case_folder(tmp_path: Path) -> Path:
    folder = tmp_path / "case"
    folder.mkdir()
    return folder


@pytest.fixture
def config(tmp_path: Path, case_folder: Path) -> AppConfig:
    cfg = AppConfig(
        case=CaseInfo(case_number="TEST-001", title="t", description="d"),
        paths=Paths(
            case_folder=case_folder,
            data_dir=tmp_path / "data",
            audio_dir=tmp_path / "data" / "audio",
            frames_dir=tmp_path / "data" / "frames",
            pages_dir=tmp_path / "data" / "pages",
        ),
    )
    cfg.root = tmp_path
    return cfg
