"""Loads and resolves config.yaml for the ingestion pipeline."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class Config:
    raw: dict
    root: Path

    @property
    def case_number(self) -> str:
        return self.raw["case"]["case_number"]

    @property
    def case_title(self) -> str:
        return self.raw["case"].get("title", "")

    @property
    def case_description(self) -> str:
        return self.raw["case"].get("description", "")

    @property
    def db(self) -> dict:
        return self.raw["database"]

    def path(self, key: str) -> Path:
        p = Path(self.raw["paths"][key])
        return p if p.is_absolute() else (self.root / p).resolve()

    @property
    def case_folder(self) -> Path:
        return self.path("case_folder")

    @property
    def data_dir(self) -> Path:
        return self.path("data_dir")

    @property
    def audio_dir(self) -> Path:
        return self.path("audio_dir")

    @property
    def frames_dir(self) -> Path:
        return self.path("frames_dir")

    @property
    def pages_dir(self) -> Path:
        return self.path("pages_dir")

    @property
    def frame_sample_rate_sec(self) -> int:
        return self.raw.get("processing", {}).get("frame_sample_rate_sec", 1)

    @property
    def fallback_window_sec(self) -> int:
        return self.raw.get("processing", {}).get("fallback_window_sec", 5)

    @property
    def audio_sample_rate_hz(self) -> int:
        return self.raw.get("processing", {}).get("audio_sample_rate_hz", 16000)

    def file_metadata(self, relative_path: str) -> dict:
        """Look up declared type/metadata for a file by its path relative to case_folder."""
        for entry in self.raw.get("files", []):
            if entry["path"] == relative_path:
                return entry
        return {}

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.audio_dir, self.frames_dir, self.pages_dir):
            d.mkdir(parents=True, exist_ok=True)


def load_config(path: str | os.PathLike = "config.yaml") -> Config:
    path = Path(path).resolve()
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return Config(raw=raw, root=path.parent)
