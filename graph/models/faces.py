"""Face detection and embedding via insightface.

insightface ships pretrained detection + ArcFace recognition as one pipeline
(`FaceAnalysis`), which is why this wraps one model rather than two: hand-
rolling a separate detector and embedder would just reimplement what the
package already does as a single ONNX graph.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from enrichment.models.base import LazyModel

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class FaceDetection:
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float
    embedding: np.ndarray  # L2-normalised, matches insightface's own convention


class FaceDetector(LazyModel):
    """insightface FaceAnalysis (detection + ArcFace recognition)."""

    def __init__(self, model_pack: str, expected_dim: int, min_confidence: float) -> None:
        super().__init__()
        self.name = f"insightface({model_pack})"
        self._model_pack = model_pack
        self._expected_dim = expected_dim
        self._min_confidence = min_confidence

    def _build(self):
        from insightface.app import FaceAnalysis

        app = FaceAnalysis(name=self._model_pack, providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=-1, det_size=(640, 640))  # ctx_id=-1: CPU
        return app

    def detect(self, image_path: Path) -> list[FaceDetection]:
        app = self.load()
        if app is None or not image_path.is_file():
            return []

        image = self._read(image_path)
        if image is None:
            return []

        try:
            faces = app.get(image)
        except Exception as exc:  # noqa: BLE001 - a bad frame must not stop the run
            log.warning("face detection failed for %s: %s", image_path.name, exc)
            return []

        detections = []
        for face in faces:
            confidence = float(getattr(face, "det_score", 1.0))
            if confidence < self._min_confidence:
                continue
            embedding = np.asarray(face.normed_embedding, dtype=np.float32)
            if embedding.shape[0] != self._expected_dim:
                log.warning(
                    "insightface returned a %d-d embedding, expected %d; skipping face",
                    embedding.shape[0], self._expected_dim,
                )
                continue
            x1, y1, x2, y2 = (float(v) for v in face.bbox)
            detections.append(
                FaceDetection(bbox=(x1, y1, x2, y2), confidence=confidence, embedding=embedding)
            )
        return detections

    @staticmethod
    def _read(image_path: Path):
        """Decode via OpenCV, which is what insightface's own examples use and
        what its FaceAnalysis.get() expects (BGR ndarray)."""
        import cv2

        image = cv2.imread(str(image_path))
        if image is None:
            log.warning("cannot decode %s for face detection", image_path.name)
        return image
