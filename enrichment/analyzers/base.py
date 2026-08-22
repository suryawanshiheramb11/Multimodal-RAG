"""The analyzer contract and the node record it works on.

Mirrors the ingestion phase's processor design: an analyzer receives one
pending node and returns a result object. It never touches the database, so it
can be tested with a fixture node and a stub registry.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import numpy as np

from ..errors import AnalyzerNotRegisteredError


@dataclass(frozen=True)
class PendingNode:
    """An evidence_node row awaiting feature extraction."""

    id: str
    node_type: str
    source_file_id: str
    source_path: Path
    source_type: str
    start_time: float | None
    end_time: float | None
    page_number: int | None
    text_content: str | None
    file_path: str | None
    metadata: dict = field(default_factory=dict)

    @property
    def frame_paths(self) -> list[Path]:
        """Frames recorded by the ingestion phase, in timestamp order."""
        frames = self.metadata.get("frames") or []
        return [Path(f["path"]) for f in frames if isinstance(f, dict) and f.get("path")]

    @property
    def audio_path(self) -> Path | None:
        raw = self.metadata.get("audio_path") or (
            self.file_path if self.node_type == "audio_track" else None
        )
        return Path(raw) if raw else None

    def representative_frame(self) -> Path | None:
        """The middle frame: most likely to show the substance of a segment,
        where the first and last often catch a transition."""
        frames = self.frame_paths
        if not frames:
            return None
        return frames[len(frames) // 2]


@dataclass
class EnrichmentResult:
    """What an analyzer extracted, ready for the repository to persist."""

    text_content: str | None = None
    text_embedding: np.ndarray | None = None
    clip_embedding: np.ndarray | None = None
    audio_embedding: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)
    #: Features that were requested but could not run, and why.
    skipped: dict[str, str] = field(default_factory=dict)

    def note_skip(self, feature: str, reason: str) -> None:
        self.skipped[feature] = reason


class NodeAnalyzer(ABC):
    """Extracts features from one kind of evidence node."""

    node_type: ClassVar[str]

    @abstractmethod
    def analyze(self, node: PendingNode) -> EnrichmentResult:
        """Run every enabled model against `node` and return the results."""


class AnalyzerRegistry:
    """Dispatches a node to the analyzer registered for its node_type."""

    def __init__(self, analyzers: list[NodeAnalyzer] | None = None) -> None:
        self._analyzers: dict[str, NodeAnalyzer] = {}
        for analyzer in analyzers or []:
            self.register(analyzer)

    def register(self, analyzer: NodeAnalyzer) -> None:
        self._analyzers[analyzer.node_type] = analyzer

    def get(self, node_type: str) -> NodeAnalyzer:
        try:
            return self._analyzers[node_type]
        except KeyError as exc:
            raise AnalyzerNotRegisteredError(
                f"no analyzer registered for node type '{node_type}'"
            ) from exc

    def supports(self, node_type: str) -> bool:
        return node_type in self._analyzers

    @property
    def supported_types(self) -> set[str]:
        return set(self._analyzers)
