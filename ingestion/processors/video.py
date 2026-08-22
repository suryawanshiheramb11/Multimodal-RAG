"""Video ingestion: normalise audio, split into segments, sample frames."""
from __future__ import annotations

import logging
from typing import ClassVar

from ..errors import MediaProcessingError
from ..media import AudioExtractor, FrameSampler, SceneSegmenter, probe_video
from ..models import EvidenceNodeDraft, MediaType, NodeType, ScannedFile
from ..workspace import Workspace
from .base import FileProcessor

log = logging.getLogger(__name__)


class VideoProcessor(FileProcessor):
    """Produces one scene_segment node per segment, each carrying its frames.

    A silent video is not an error: the audio track is optional and its absence
    is recorded on the node rather than failing the file.
    """

    media_type: ClassVar[MediaType] = MediaType.VIDEO

    def __init__(
        self,
        workspace: Workspace,
        segmenter: SceneSegmenter,
        sampler: FrameSampler,
        audio_extractor: AudioExtractor,
    ) -> None:
        self._workspace = workspace
        self._segmenter = segmenter
        self._sampler = sampler
        self._audio_extractor = audio_extractor

    def process(self, source: ScannedFile) -> list[EvidenceNodeDraft]:
        info = probe_video(source.path)
        segments, strategy = self._segmenter.segment(source.path, info.duration_sec)
        audio_path = self._extract_audio(source)
        frames = self._sampler.sample(
            source.path, segments, self._workspace.frames_dir(source)
        )

        log.info(
            "video %s: %.1fs, %d segments (%s), %d frames, audio=%s",
            source.file_name, info.duration_sec, len(segments), strategy,
            sum(len(v) for v in frames.values()), "yes" if audio_path else "none",
        )

        return [
            EvidenceNodeDraft(
                node_type=NodeType.SCENE_SEGMENT,
                start_time=segment.start,
                end_time=segment.end,
                file_path=str(audio_path) if audio_path else None,
                metadata={
                    "segment_index": segment.index,
                    "segmentation": strategy,
                    "duration_sec": round(segment.duration, 3),
                    "width": info.width,
                    "height": info.height,
                    "codec": info.codec,
                    "audio_path": str(audio_path) if audio_path else None,
                    "frames": [
                        {"timestamp": round(f.timestamp, 3), "path": str(f.path)}
                        for f in frames.get(segment.index, [])
                    ],
                },
            )
            for segment in segments
        ]

    def _extract_audio(self, source: ScannedFile):
        """Return the extracted audio path, or None when the video is silent."""
        try:
            return self._audio_extractor.extract(
                source.path, self._workspace.audio_path(source)
            )
        except MediaProcessingError as exc:
            log.warning("no audio extracted from %s: %s", source.file_name, exc)
            return None
