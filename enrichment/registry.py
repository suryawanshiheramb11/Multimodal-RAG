"""Model registry: one place that owns every model instance.

Models are shared across analyzers and loaded at most once per run, which is
what makes a multi-hundred-node case affordable on CPU (or GPU via MPS/CUDA).
Availability is reported up front so the run log states plainly which features are live.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from enrichment.optimization import get_device

from .config import EnrichmentSettings
from .models import (
    AudioEventClassifier,
    Captioner,
    ClipEncoder,
    LazyModel,
    ObjectDetector,
    OcrReader,
    TextEncoder,
    Transcriber,
)

log = logging.getLogger(__name__)


@dataclass
class ModelRegistry:
    """Holds every model wrapper. Construction is cheap; loading is deferred."""

    transcriber: Transcriber
    audio_events: AudioEventClassifier
    clip: ClipEncoder
    detector: ObjectDetector
    captioner: Captioner
    ocr: OcrReader
    text_encoder: TextEncoder

    @classmethod
    def build(cls, settings: EnrichmentSettings) -> ModelRegistry:
        names = settings.models
        device = get_device()  # Auto-detect MPS/CUDA/CPU
        log.info("GPU/device: %s", device)

        return cls(
            transcriber=Transcriber(names.asr, settings.device, settings.compute_type),
            audio_events=AudioEventClassifier(
                names.audio_events, settings.audio_embedding_dim, device=device
            ),
            clip=ClipEncoder(names.clip, settings.clip_embedding_dim, device=device),
            detector=ObjectDetector(
                names.detector, settings.detection_confidence, device=device
            ),
            captioner=Captioner(
                names.captioner,
                settings.ollama_host,
                settings.ollama_timeout_sec,
                keep_alive=settings.ollama_keep_alive,
                max_tokens=settings.caption_max_tokens,
                max_image_side=settings.caption_max_side,
            ),
            ocr=OcrReader(settings.ocr_language, settings.ocr_max_side),
            text_encoder=TextEncoder(
                names.text_encoder, settings.text_embedding_dim, device=device
            ),
        )

    def all_models(self) -> dict[str, LazyModel]:
        return {
            "asr": self.transcriber,
            "audio_events": self.audio_events,
            "clip": self.clip,
            "detection": self.detector,
            "caption": self.captioner,
            "ocr": self.ocr,
            "text_encoder": self.text_encoder,
        }

    def availability(self) -> dict[str, str]:
        """Force-load everything and report status per feature.

        Called once before processing so failures surface as a single readable
        table instead of being scattered through the per-node warnings.
        """
        report: dict[str, str] = {}
        for key, model in self.all_models().items():
            report[key] = "ready" if model.available else (
                model.unavailable_reason or "unavailable"
            )
        return report
