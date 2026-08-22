"""Text extraction from images with PaddleOCR.

PaddleOCR's API changed between 2.x and 3.x — `use_angle_cls` became
`use_textline_orientation`, and `.ocr()` gave way to `.predict()` with a
different result shape. Both are handled so the pipeline works against
whichever version is installed.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from .base import LazyModel

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class OcrResult:
    text: str
    line_count: int
    mean_confidence: float | None


class OcrReader(LazyModel):
    """PaddleOCR text detection + recognition."""

    def __init__(self, language: str = "en") -> None:
        super().__init__()
        self.name = f"paddleocr({language})"
        self._language = language
        self._uses_legacy_api = False

    def _build(self):
        from paddleocr import PaddleOCR

        try:
            model = PaddleOCR(use_textline_orientation=True, lang=self._language)
        except (TypeError, ValueError):
            # PaddleOCR 2.x spelling.
            model = PaddleOCR(use_angle_cls=True, lang=self._language)
            self._uses_legacy_api = True
        return model

    def read(self, image_path: Path) -> OcrResult | None:
        model = self.load()
        if model is None or not image_path.is_file():
            return None

        try:
            lines = self._run(model, image_path)
        except Exception as exc:  # noqa: BLE001 - OCR faults must not stop the run
            log.warning("OCR failed for %s: %s", image_path.name, exc)
            return None

        if not lines:
            return OcrResult(text="", line_count=0, mean_confidence=None)

        texts = [text for text, _ in lines]
        confidences = [score for _, score in lines if score is not None]
        return OcrResult(
            text="\n".join(texts),
            line_count=len(texts),
            mean_confidence=(
                round(sum(confidences) / len(confidences), 4) if confidences else None
            ),
        )

    def _run(self, model, image_path: Path) -> list[tuple[str, float | None]]:
        if not self._uses_legacy_api and hasattr(model, "predict"):
            return self._parse_v3(model.predict(str(image_path)))
        return self._parse_v2(model.ocr(str(image_path)))

    @staticmethod
    def _parse_v3(results) -> list[tuple[str, float | None]]:
        """3.x returns dict-like records with parallel text/score lists."""
        lines: list[tuple[str, float | None]] = []
        for record in results or []:
            data = record if isinstance(record, dict) else getattr(record, "json", {})
            data = data.get("res", data) if isinstance(data, dict) else {}
            texts = data.get("rec_texts") or []
            scores = data.get("rec_scores") or [None] * len(texts)
            lines.extend(zip(texts, scores, strict=False))
        return lines

    @staticmethod
    def _parse_v2(results) -> list[tuple[str, float | None]]:
        """2.x returns [[ [box, (text, score)], ... ]] per image."""
        lines: list[tuple[str, float | None]] = []
        for page in results or []:
            for entry in page or []:
                if not entry or len(entry) < 2:
                    continue
                payload = entry[1]
                if isinstance(payload, (list, tuple)) and payload:
                    lines.append((str(payload[0]), float(payload[1]) if len(payload) > 1 else None))
        return lines
