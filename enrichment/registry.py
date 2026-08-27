"""Model registry: one place that owns every model instance.

Models are shared across analyzers and loaded at most once per run, which is
what makes a multi-hundred-node case affordable on CPU (or GPU via MPS/CUDA).
Availability is reported up front so the run log states plainly which features are live.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

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
from .models.base import get_device

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
            transcriber=Transcriber(
                names.asr,
                settings.device,
                settings.compute_type,
                settings.asr_cpu_threads,
                settings.asr_batch_size,
            ),
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
            ocr=OcrReader(
                settings.ocr_language,
                settings.ocr_max_side,
                settings.ocr_det_model,
                settings.ocr_rec_model,
            ),
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

        The loads run concurrently. They are independent and mostly spent
        waiting — reading checkpoints off disk, spawning the OCR process,
        warming ollama's weights — so serialising them just adds up to a long
        silence before the first node. `LazyModel.load` is locked per model, so
        each is still built exactly once.
        """
        models = self.all_models()
        report: dict[str, str] = {}

        def status(model: LazyModel) -> str:
            return "ready" if model.available else (
                model.unavailable_reason or "unavailable"
            )

        with ThreadPoolExecutor(
            max_workers=len(models), thread_name_prefix="model-load"
        ) as pool:
            futures = {key: pool.submit(status, model) for key, model in models.items()}
            for key, future in futures.items():
                try:
                    report[key] = future.result()
                except Exception as exc:  # noqa: BLE001 - mirrors LazyModel policy
                    report[key] = f"{type(exc).__name__}: {exc}"
        return report
