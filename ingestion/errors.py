"""Exception hierarchy for the ingestion pipeline.

Callers catch `IngestionError` to handle any pipeline failure; the specific
subclasses let the pipeline decide whether to skip one file or abort the run.
"""
from __future__ import annotations


class IngestionError(Exception):
    """Base class for every error raised by this package."""


class ConfigError(IngestionError):
    """config.yaml is missing, malformed, or fails validation."""


class SecurityError(IngestionError):
    """A file or path violated a containment or safety rule.

    Raised for path traversal, symlink escapes, and declared/detected media
    type conflicts. These are refusals, not crashes: the file is quarantined
    from processing and the run continues.
    """


class ResourceLimitError(IngestionError):
    """A file exceeded a configured resource limit (size, pages, duration).

    Guards against decompression bombs and pathological inputs that would
    otherwise exhaust memory or disk.
    """


class MediaProcessingError(IngestionError):
    """An external tool (ffmpeg, PyAV, PyMuPDF) failed on a file."""


class UnsupportedMediaTypeError(IngestionError):
    """No processor is registered for the file's media type."""
