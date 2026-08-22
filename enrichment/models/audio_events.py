"""Audio event tagging with AST (Audio Spectrogram Transformer).

Produces AudioSet labels ("Gunshot", "Screaming", "Speech") plus a pooled
hidden-state embedding for the audio vector column.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .base import LazyModel

log = logging.getLogger(__name__)

#: AST was trained on 16 kHz audio, which is also what the ingestion phase
#: normalises every track to.
SAMPLE_RATE = 16_000


@dataclass(frozen=True)
class AudioEvent:
    label: str
    probability: float


@dataclass(frozen=True)
class AudioAnalysis:
    events: list[AudioEvent]
    embedding: np.ndarray


class AudioEventClassifier(LazyModel):
    """MIT/ast-finetuned-audioset — multi-label tagging over 527 classes."""

    def __init__(self, model_name: str, expected_dim: int) -> None:
        super().__init__()
        self.name = f"ast({model_name})"
        self._model_name = model_name
        self._expected_dim = expected_dim
        self._extractor = None
        self._torch = None

    def _build(self):
        import torch
        from transformers import ASTForAudioClassification, AutoFeatureExtractor

        model = ASTForAudioClassification.from_pretrained(self._model_name)
        model.eval()
        if model.config.hidden_size != self._expected_dim:
            raise ValueError(
                f"{self._model_name} has hidden size {model.config.hidden_size} "
                f"but the schema column expects {self._expected_dim}"
            )
        self._extractor = AutoFeatureExtractor.from_pretrained(self._model_name)
        self._torch = torch
        return model

    def analyse(
        self, audio_path: Path, start: float | None = None, end: float | None = None,
        top_k: int = 5,
    ) -> AudioAnalysis | None:
        """Tag the audio in [start, end), or the whole file when unbounded."""
        model = self.load()
        if model is None:
            return None

        waveform = self._load_waveform(audio_path, start, end)
        if waveform is None or waveform.size == 0:
            return None

        try:
            inputs = self._extractor(
                waveform, sampling_rate=SAMPLE_RATE, return_tensors="pt"
            )
            with self._torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)

            # AudioSet is multi-label, so each class gets an independent
            # sigmoid rather than a softmax across classes.
            probabilities = self._torch.sigmoid(outputs.logits)[0]
            scores, indices = probabilities.topk(min(top_k, probabilities.shape[-1]))

            events = [
                AudioEvent(label=model.config.id2label[int(i)], probability=round(float(s), 4))
                for s, i in zip(scores, indices, strict=True)
            ]
            # Mean-pool the final layer over time to get one vector per clip.
            embedding = outputs.hidden_states[-1].mean(dim=1)[0].cpu().numpy()
        except Exception as exc:  # noqa: BLE001 - one clip must not stop the run
            log.warning("audio tagging failed for %s: %s", audio_path.name, exc)
            return None

        return AudioAnalysis(events=events, embedding=embedding.astype(np.float32))

    @staticmethod
    def _load_waveform(
        audio_path: Path, start: float | None, end: float | None
    ) -> np.ndarray | None:
        import librosa

        if not audio_path.is_file():
            log.warning("audio file missing for tagging: %s", audio_path)
            return None

        offset = max(start or 0.0, 0.0)
        duration = (end - offset) if (end is not None and end > offset) else None
        try:
            waveform, _ = librosa.load(
                str(audio_path), sr=SAMPLE_RATE, mono=True,
                offset=offset, duration=duration,
            )
        except Exception as exc:  # noqa: BLE001 - unreadable audio is not fatal
            log.warning("cannot read %s: %s", audio_path.name, exc)
            return None
        return waveform
