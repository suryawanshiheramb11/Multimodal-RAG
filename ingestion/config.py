"""Configuration loading and validation.

Structure and non-secret values come from config.yaml; credentials come from
the environment only. Validation is delegated to pydantic rather than
hand-rolled getters, so a malformed config fails loudly at startup with a
field-level error instead of an AttributeError deep in a processor.
"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, SecretStr, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .errors import ConfigError
from .models import MediaType

ENV_PREFIX = "EVIDENCE_DB_"


class DatabaseSettings(BaseSettings):
    """Postgres connection settings.

    The password is never read from config.yaml — only from
    ``EVIDENCE_DB_PASSWORD`` (or libpq's own ``PGPASSWORD``/``~/.pgpass``) — so
    that the committed config file can never carry a credential. It is held as
    a SecretStr so it cannot leak through logs or tracebacks.
    """

    model_config = SettingsConfigDict(env_prefix=ENV_PREFIX, extra="ignore")

    host: str = "localhost"
    port: int = Field(default=5432, ge=1, le=65535)
    dbname: str = "evidence_db"
    user: str = "postgres"
    password: SecretStr | None = None
    connect_timeout: int = Field(default=10, ge=1, le=120)
    sslmode: str = "prefer"

    @classmethod
    def build(cls, yaml_section: dict | None) -> DatabaseSettings:
        """Environment wins over YAML.

        pydantic-settings ranks init kwargs above env vars, which is the
        opposite of what we want, so YAML values are only passed for fields the
        environment does not already define.
        """
        section = dict(yaml_section or {})
        section.pop("password", None)  # never honour a password from YAML
        overrides = {
            key: value
            for key, value in section.items()
            if f"{ENV_PREFIX}{key.upper()}" not in os.environ
        }
        return cls(**overrides)

    def connect_kwargs(self) -> dict:
        return {
            "host": self.host,
            "port": self.port,
            "dbname": self.dbname,
            "user": self.user,
            "password": self.password.get_secret_value() if self.password else None,
            "connect_timeout": self.connect_timeout,
            "sslmode": self.sslmode,
        }

    def safe_dsn(self) -> str:
        """Connection description with no credential in it, safe to log."""
        return f"postgresql://{self.user}@{self.host}:{self.port}/{self.dbname}"


class Limits(BaseModel):
    """Hard ceilings applied to untrusted input.

    Every one of these exists to stop a single malicious or corrupt file from
    exhausting memory, disk, or wall-clock time.
    """

    model_config = {"extra": "forbid"}

    max_file_bytes: int = Field(default=2 * 1024**3, gt=0)
    max_video_seconds: float = Field(default=7200.0, gt=0)
    max_frames_per_file: int = Field(default=3600, gt=0)
    max_pdf_pages: int = Field(default=2000, gt=0)
    max_page_text_chars: int = Field(default=200_000, gt=0)
    max_image_pixels: int = Field(default=100_000_000, gt=0)
    ffmpeg_timeout_seconds: int = Field(default=900, gt=0)
    scene_detect_timeout_seconds: int = Field(default=900, gt=0)


class Processing(BaseModel):
    model_config = {"extra": "forbid"}

    frame_sample_rate_sec: float = Field(default=1.0, gt=0)
    fallback_window_sec: float = Field(default=5.0, gt=0)
    audio_sample_rate_hz: int = Field(default=16000, gt=0)
    pdf_render_zoom: float = Field(default=2.0, gt=0, le=8.0)
    scene_detect_threshold: float = Field(default=27.0, gt=0)


class Paths(BaseModel):
    model_config = {"extra": "forbid"}

    case_folder: Path = Path("./sample_case")
    data_dir: Path = Path("./data")
    audio_dir: Path = Path("./data/audio")
    frames_dir: Path = Path("./data/frames")
    pages_dir: Path = Path("./data/pages")


class CaseInfo(BaseModel):
    model_config = {"extra": "forbid"}

    case_number: str = Field(min_length=1, max_length=128)
    title: str | None = None
    description: str | None = None


class FileEntry(BaseModel):
    """A declared evidence file: its path relative to the case folder plus
    whatever provenance metadata the investigator recorded for it."""

    model_config = {"extra": "allow"}

    path: str = Field(min_length=1)
    type: MediaType | None = None
    author: str | None = None
    created_date: datetime | None = None

    @field_validator("path")
    @classmethod
    def _reject_absolute_or_parent(cls, value: str) -> str:
        candidate = Path(value)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError("must be a relative path inside the case folder")
        return value


class AppConfig(BaseModel):
    model_config = {"extra": "forbid"}

    case: CaseInfo
    paths: Paths = Paths()
    processing: Processing = Processing()
    limits: Limits = Limits()
    files: list[FileEntry] = Field(default_factory=list)

    # Populated after load(); not part of the YAML document.
    root: Path = Field(default=Path("."), exclude=True)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings, exclude=True)

    def resolve(self, path: Path) -> Path:
        """Make a configured path absolute, relative to the config file."""
        return path if path.is_absolute() else (self.root / path).resolve()

    @property
    def case_folder(self) -> Path:
        return self.resolve(self.paths.case_folder)

    def declared_entry(self, relative_path: str) -> FileEntry | None:
        """Look up the config entry for a file by its case-folder-relative path.

        Compared with PurePath so that ``video/a.mp4`` and ``video\\a.mp4``
        match the same entry regardless of how the config was written.
        """
        target = Path(relative_path).as_posix()
        for entry in self.files:
            if Path(entry.path).as_posix() == target:
                return entry
        return None


def load_config(path: str | os.PathLike = "config.yaml") -> AppConfig:
    """Read, validate, and return the application config.

    Raises ConfigError with the offending field on any validation failure.
    """
    config_path = Path(path).resolve()
    try:
        raw = yaml.safe_load(config_path.read_text()) or {}
    except FileNotFoundError as exc:
        raise ConfigError(f"config file not found: {config_path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"config file is not valid YAML: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigError("config file must contain a YAML mapping at the top level")

    db_section = raw.pop("database", {})
    try:
        config = AppConfig(**raw)
        config.database = DatabaseSettings.build(db_section)
    except ValidationError as exc:
        raise ConfigError(f"invalid configuration:\n{exc}") from exc

    config.root = config_path.parent
    return config
