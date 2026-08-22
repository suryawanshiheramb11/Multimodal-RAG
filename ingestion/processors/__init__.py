"""Media processors and the registry that dispatches to them."""
from __future__ import annotations

from ..config import AppConfig
from ..media import AudioExtractor, FrameSampler, SceneSegmenter
from ..workspace import Workspace
from .audio import AudioProcessor
from .base import FileProcessor, ProcessorRegistry
from .image import ImageProcessor
from .pdf import PdfProcessor
from .video import VideoProcessor

__all__ = [
    "AudioProcessor",
    "FileProcessor",
    "ImageProcessor",
    "PdfProcessor",
    "ProcessorRegistry",
    "VideoProcessor",
    "build_registry",
]


def build_registry(config: AppConfig, workspace: Workspace) -> ProcessorRegistry:
    """Composition root: wire every processor with its collaborators.

    This is the one place that knows which concrete classes exist. Everything
    downstream depends on the FileProcessor abstraction, so adding a
    DocProcessor means adding one line here and nothing else.
    """
    limits = config.limits
    processing = config.processing

    audio_extractor = AudioExtractor(
        sample_rate_hz=processing.audio_sample_rate_hz,
        timeout_sec=limits.ffmpeg_timeout_seconds,
    )
    segmenter = SceneSegmenter(
        threshold=processing.scene_detect_threshold,
        fallback_window_sec=processing.fallback_window_sec,
        max_duration_sec=limits.max_video_seconds,
    )
    sampler = FrameSampler(
        interval_sec=processing.frame_sample_rate_sec,
        max_frames=limits.max_frames_per_file,
    )

    return ProcessorRegistry(
        [
            VideoProcessor(workspace, segmenter, sampler, audio_extractor),
            AudioProcessor(workspace, audio_extractor),
            PdfProcessor(
                workspace,
                zoom=processing.pdf_render_zoom,
                max_pages=limits.max_pdf_pages,
                max_text_chars=limits.max_page_text_chars,
                max_pixels=limits.max_image_pixels,
            ),
            ImageProcessor(max_pixels=limits.max_image_pixels),
        ]
    )
