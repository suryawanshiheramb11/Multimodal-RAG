"""Object detection with YOLO (GPU-accelerated via MPS/CUDA)."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import torch

from .base import LazyModel, get_device

log = logging.getLogger(__name__)

#: COCO classes that matter for a violence investigation. Recorded separately
#: so a reviewer can filter on them without re-running detection.
NOTABLE_CLASSES = frozenset({"knife", "scissors", "baseball bat", "gun", "pistol", "rifle"})


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    #: xyxy in pixels, matching Ultralytics' native box format.
    bbox: tuple[float, float, float, float]
    frame_path: str


class ObjectDetector(LazyModel):
    """Ultralytics YOLO over sampled frames (GPU-accelerated)."""

    def __init__(self, weights: str, confidence: float, device: torch.device | None = None) -> None:
        super().__init__()
        self.name = f"yolo({weights})"
        self._weights = weights
        self._confidence = confidence
        self._device = device or get_device()

    def _build(self):
        from ultralytics import YOLO

        model = YOLO(self._weights)
        # Move model to GPU/MPS if available; YOLO will use device parameter in predict
        log.info("YOLO on device %s", self._device)
        return model

    def detect(self, image_paths: list[Path]) -> list[Detection]:
        """Detect objects across frames (batched inference on GPU). Returns [] when unavailable."""
        model = self.load()
        if model is None or not image_paths:
            return []

        existing = [p for p in image_paths if p.is_file()]
        if not existing:
            return []

        try:
            # Batch prediction: YOLO processes all images in one forward pass
            # device parameter routes computation to GPU/MPS/CPU
            results = model.predict(
                [str(p) for p in existing],
                conf=self._confidence,
                device=self._device,  # Route to GPU/MPS
                verbose=False,
                imgsz=640,  # Fixed size for GPU efficiency
            )
        except Exception as exc:  # noqa: BLE001 - detector faults are not fatal
            log.warning("detection failed: %s", exc)
            return []

        detections: list[Detection] = []
        for path, result in zip(existing, results, strict=True):
            names = result.names
            for box in result.boxes:
                detections.append(
                    Detection(
                        label=names[int(box.cls)],
                        confidence=round(float(box.conf), 4),
                        bbox=tuple(round(float(v), 2) for v in box.xyxy[0].tolist()),
                        frame_path=str(path),
                    )
                )
        return detections

    @staticmethod
    def summarise(detections: list[Detection]) -> dict:
        """Counts and peak confidence per label, plus any notable weapons."""
        summary: dict[str, dict] = {}
        for detection in detections:
            entry = summary.setdefault(
                detection.label, {"count": 0, "max_confidence": 0.0}
            )
            entry["count"] += 1
            entry["max_confidence"] = max(entry["max_confidence"], detection.confidence)

        return {
            "labels": summary,
            "total": len(detections),
            "notable": sorted(set(summary) & NOTABLE_CLASSES),
        }
