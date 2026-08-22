"""Tests for configuration loading, validation, and credential handling."""
from __future__ import annotations

import textwrap

import pytest

from ingestion.config import DatabaseSettings, load_config
from ingestion.errors import ConfigError

_MINIMAL = """
case:
  case_number: "CASE-1"
paths:
  case_folder: "./evidence"
"""


def _write(tmp_path, *parts: str):
    """Write a config file from fragments, dedenting each one separately.

    Dedenting the concatenation would be a no-op, since the un-indented base
    fragment makes the common prefix empty.
    """
    path = tmp_path / "config.yaml"
    path.write_text("".join(textwrap.dedent(part) for part in parts))
    return path


class TestCredentials:
    def test_password_in_yaml_is_ignored(self, tmp_path, monkeypatch):
        """A credential committed to config.yaml must never be honoured."""
        monkeypatch.delenv("EVIDENCE_DB_PASSWORD", raising=False)
        path = _write(tmp_path, _MINIMAL, """
        database:
          host: "db.internal"
          password: "hunter2"
        """)

        config = load_config(path)

        assert config.database.password is None
        assert config.database.host == "db.internal"

    def test_password_comes_from_the_environment(self, tmp_path, monkeypatch):
        monkeypatch.setenv("EVIDENCE_DB_PASSWORD", "from-env")
        config = load_config(_write(tmp_path, _MINIMAL))

        assert config.database.password.get_secret_value() == "from-env"

    def test_environment_overrides_yaml(self, tmp_path, monkeypatch):
        monkeypatch.setenv("EVIDENCE_DB_HOST", "env-host")
        path = _write(tmp_path, _MINIMAL, """
        database:
          host: "yaml-host"
        """)

        assert load_config(path).database.host == "env-host"

    def test_secret_does_not_leak_through_repr_or_dsn(self, monkeypatch):
        monkeypatch.setenv("EVIDENCE_DB_PASSWORD", "topsecret")
        settings = DatabaseSettings()

        assert "topsecret" not in repr(settings)
        assert "topsecret" not in str(settings)
        assert "topsecret" not in settings.safe_dsn()
        # ...but the real value is still available to the driver.
        assert settings.connect_kwargs()["password"] == "topsecret"


class TestValidation:
    def test_missing_case_number_is_rejected(self, tmp_path):
        with pytest.raises(ConfigError, match="case"):
            load_config(_write(tmp_path, "paths:\n  case_folder: './x'\n"))

    def test_unknown_key_is_rejected(self, tmp_path):
        """extra='forbid' turns a typo into a startup error, not silence."""
        with pytest.raises(ConfigError):
            load_config(_write(tmp_path, _MINIMAL + "\nprocessing:\n  frame_rate: 1\n"))

    def test_absolute_file_path_is_rejected(self, tmp_path):
        with pytest.raises(ConfigError, match="relative path"):
            load_config(_write(tmp_path, _MINIMAL, """
            files:
              - path: "/etc/passwd"
                type: "pdf"
            """))

    def test_parent_traversal_in_file_path_is_rejected(self, tmp_path):
        with pytest.raises(ConfigError, match="relative path"):
            load_config(_write(tmp_path, _MINIMAL, """
            files:
              - path: "../../etc/passwd"
                type: "pdf"
            """))

    def test_negative_limit_is_rejected(self, tmp_path):
        with pytest.raises(ConfigError):
            load_config(_write(tmp_path, _MINIMAL, """
            limits:
              max_pdf_pages: -1
            """))

    def test_missing_file_reports_clearly(self, tmp_path):
        with pytest.raises(ConfigError, match="not found"):
            load_config(tmp_path / "absent.yaml")

    def test_malformed_yaml_reports_clearly(self, tmp_path):
        with pytest.raises(ConfigError, match="not valid YAML"):
            load_config(_write(tmp_path, "case: [unclosed\n"))


class TestLookup:
    def test_declared_entry_is_found_by_relative_path(self, tmp_path):
        config = load_config(_write(tmp_path, _MINIMAL, """
        files:
          - path: "video/clip.mp4"
            type: "video"
            author: "Det. Ruiz"
        """))

        entry = config.declared_entry("video/clip.mp4")

        assert entry is not None
        assert entry.author == "Det. Ruiz"
        assert config.declared_entry("video/other.mp4") is None
