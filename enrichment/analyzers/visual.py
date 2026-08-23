"""Shared visual feature extraction for any node backed by images.

Video segments and standalone images need exactly the same work — violence
scoring, detection, captioning, OCR, and a CLIP image vector — so it lives here
once and both analyzers compose it rather than inheriting from each other.
"""
from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..config import EnrichmentSettings
from ..registry import ModelRegistry
from .base import EnrichmentResult

log = logging.getLogger(__name__)


@dataclass
class VisualFeatures:
    caption: str | None = None
    ocr_text: str | None = None
    clip_embedding: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)
    #: Guards `metadata` and the caller's `EnrichmentResult` while the
    #: offloaded stages write into them from their own threads.
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


class VisualExtractor:
    """Runs the image-based models over a node's frames."""

    def __init__(self, registry: ModelRegistry, settings: EnrichmentSettings) -> None:
        self._registry = registry
        self._settings = settings

    def extract(
        self, frames: list[Path], representative: Path | None, result: EnrichmentResult
    ) -> VisualFeatures:
        """Analyse `frames`, using `representative` for the single-image models.

        Captioning and OCR run on one frame rather than all of them: a
        vision-language call per frame would dominate the runtime of the whole
        phase for almost no extra signal within a single scene.
        """
        features = VisualFeatures()
        existing = [f for f in frames if f.is_file()]
        if not existing:
            result.note_skip("visual", "no frames available on disk")
            return features

        sampled = self._subsample(existing)
        features.metadata["frames_analyzed"] = len(sampled)
        features.metadata["frames_available"] = len(existing)

        target = representative if (representative and representative.is_file()) else sampled[0]
        features.metadata["representative_frame"] = str(target)

        if self._settings.parallel_stages:
            # Captioning is an HTTP call to ollama and OCR is a round-trip to
            # the PaddleOCR worker process: both sit blocked on something
            # outside this interpreter, so running them alongside the in-process
            # torch models is close to free. OCR dominates a node, so the CLIP
            # and YOLO passes now finish inside its wait instead of after it.
            with ThreadPoolExecutor(max_workers=2, thread_name_prefix="visual") as pool:
                offloaded = [
                    pool.submit(self._caption, target, features, result),
                    pool.submit(self._read_text, target, features, result),
                ]
                # torch stays on this thread: MPS does not want callers from
                # several threads at once.
                self._score_violence(sampled, features, result)
                self._detect_objects(sampled, features, result)
                for future in offloaded:
                    future.result()
        else:
            self._score_violence(sampled, features, result)
            self._detect_objects(sampled, features, result)
            self._caption(target, features, result)
            self._read_text(target, features, result)

        self._embed_image(target, features, result)

        return features

    def _subsample(self, frames: list[Path]) -> list[Path]:
        """Evenly spread the frame budget across the node's timeline."""
        limit = self._settings.max_frames_analyzed
        if len(frames) <= limit:
            return frames
        step = len(frames) / limit
        return [frames[int(i * step)] for i in range(limit)]

    def _score_violence(
        self, frames: list[Path], features: VisualFeatures, result: EnrichmentResult
    ) -> None:
        if not self._settings.enable_violence:
            result.note_skip("violence", "disabled in config")
            return

        clip = self._registry.clip
        if not clip.available:
            result.note_skip("violence", clip.unavailable_reason or "CLIP unavailable")
            return

        score = clip.score_violence(
            frames, self._settings.violence_prompts, self._settings.violence_prompt_count
        )
        if score is None:
            result.note_skip("violence", "scoring produced no result")
            return

        with features.lock:
            features.metadata["violence"] = {
                "score": round(score.score, 4),
                "frames_scored": score.frames_scored,
                "prompt_probabilities": score.per_prompt,
                "violent_prompts": self._settings.violent_prompts,
            }

    def _detect_objects(
        self, frames: list[Path], features: VisualFeatures, result: EnrichmentResult
    ) -> None:
        if not self._settings.enable_detection:
            result.note_skip("detection", "disabled in config")
            return

        detector = self._registry.detector
        if not detector.available:
            result.note_skip("detection", detector.unavailable_reason or "YOLO unavailable")
            return

        detections = detector.detect(frames)
        if detections:
            labels = sorted(set(d.label for d in detections))
            log.info("detected %d object(s): %s", len(detections), ", ".join(labels[:5]))
        with features.lock:
            features.metadata["detections"] = {
                **detector.summarise(detections),
                "boxes": [
                    {
                        "label": d.label,
                        "confidence": d.confidence,
                        "bbox": list(d.bbox),
                        "frame": d.frame_path,
                    }
                    for d in detections
                ],
            }

    def _caption(
        self, frame: Path, features: VisualFeatures, result: EnrichmentResult
    ) -> None:
        if not self._settings.enable_caption:
            result.note_skip("caption", "disabled in config")
            return

        captioner = self._registry.captioner
        if not captioner.available:
            result.note_skip("caption", captioner.unavailable_reason or "captioner unavailable")
            return

        caption = captioner.caption(frame, self._settings.caption_prompt)
        with features.lock:
            if caption:
                features.caption = caption
                features.metadata["caption"] = caption
                log.info("captioned: %s", caption[:80] + ("…" if len(caption) > 80 else ""))
            else:
                result.note_skip("caption", "model returned no text")

    def _read_text(
        self, frame: Path, features: VisualFeatures, result: EnrichmentResult
    ) -> None:
        if not self._settings.enable_ocr:
            result.note_skip("ocr", "disabled in config")
            return

        reader = self._registry.ocr
        if not reader.available:
            result.note_skip("ocr", reader.unavailable_reason or "PaddleOCR unavailable")
            return

        ocr = reader.read(frame)
        with features.lock:
            if ocr is None:
                result.note_skip("ocr", "OCR produced no result")
                return

            features.ocr_text = ocr.text or None
            if ocr.text:
                log.info("ocr read %d line(s): %s", ocr.line_count, ocr.text[:60])
            features.metadata["ocr"] = {
                "text": ocr.text,
                "line_count": ocr.line_count,
                "mean_confidence": ocr.mean_confidence,
            }

    def _embed_image(
        self, frame: Path, features: VisualFeatures, result: EnrichmentResult
    ) -> None:
        clip = self._registry.clip
        if not clip.available:
            result.note_skip("clip_embedding", clip.unavailable_reason or "CLIP unavailable")
            return

        embedding = clip.embed_image(frame)
        if embedding is None:
            result.note_skip("clip_embedding", "encoder produced no vector")
            return
        features.clip_embedding = embedding
