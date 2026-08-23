"""Speaker diarization + embedding via pyannote.audio.

Turns are the diarizer's own segmentation of "who spoke when"; the embedding
attached to each turn is not re-computed per turn (a one- or two-second clip
embeds noisily) but is the diarizer's own per-speaker centroid — pyannote's
community pipeline already produces exactly one high-quality embedding per
detected speaker per file, in `DiarizeOutput.speaker_embeddings`, ordered to
match `speaker_diarization.labels()`. Every turn belonging to a given local
speaker label gets that speaker's embedding.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from enrichment.models.base import LazyModel

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SpeakerTurn:
    start: float
    end: float
    #: Local to the one audio file just diarized (e.g. 'SPEAKER_00') —
    #: meaningless across files until case-wide clustering ties turns
    #: together into a voice_cluster.
    speaker_label: str
    embedding: np.ndarray  # L2-normalised, matching every other embedding here


class SpeakerDiarizer(LazyModel):
    """pyannote.audio's community diarization pipeline: turns + per-speaker
    embeddings, in one pass.

    Every pretrained pyannote checkpoint is gated on Hugging Face — the
    weights are free, but a user must visit the model page, accept its terms,
    and generate a token before code can load it. That is not a bug to work
    around; it is reported through the same available/unavailable_reason
    contract every model in this pipeline uses, so a case run without a token
    configured still completes — it just has no identities.
    """

    def __init__(self, checkpoint: str, expected_dim: int, auth_token: str | None) -> None:
        super().__init__()
        self.name = f"pyannote({checkpoint})"
        self._checkpoint = checkpoint
        self._expected_dim = expected_dim
        self._auth_token = auth_token

    def _build(self):
        if not self._auth_token:
            raise RuntimeError(
                "no Hugging Face token configured (set HF_TOKEN or PYANNOTE_AUTH_TOKEN); "
                f"'{self._checkpoint}' is a gated model, so a token is required even after "
                f"accepting its terms at https://huggingface.co/{self._checkpoint}"
            )

        from pyannote.audio import Pipeline

        pipeline = Pipeline.from_pretrained(self._checkpoint, token=self._auth_token)
        if pipeline is None:
            raise RuntimeError(
                f"'{self._checkpoint}' could not be loaded — the token may not have access; "
                f"accept the model's terms at https://huggingface.co/{self._checkpoint}"
            )

        # Verified rather than assumed: a silent embedding-width mismatch
        # would corrupt every voice_cluster comparison downstream. See the
        # phase 7 schema migration for how 256 was itself determined.
        actual_dim = getattr(getattr(pipeline, "_embedding", None), "dimension", None)
        if actual_dim is not None and actual_dim != self._expected_dim:
            raise ValueError(
                f"{self._checkpoint} embeds at {actual_dim}-d but the schema column "
                f"expects {self._expected_dim}"
            )
        return pipeline

    def diarize(self, audio_path: Path) -> list[SpeakerTurn]:
        pipeline = self.load()
        if pipeline is None or not audio_path.is_file():
            return []

        try:
            output = pipeline(str(audio_path))
        except Exception as exc:  # noqa: BLE001 - a bad/short audio file must not stop the run
            log.warning("diarization failed for %s: %s", audio_path.name, exc)
            return []

        # `legacy=True` pipelines return a bare Annotation with no embeddings;
        # guarding for that keeps this usable even if a caller configures one.
        diarization = getattr(output, "speaker_diarization", output)
        embeddings = getattr(output, "speaker_embeddings", None)
        if embeddings is None or len(embeddings) == 0:
            return []

        by_label: dict[str, np.ndarray] = {}
        for label, vector in zip(diarization.labels(), embeddings, strict=False):
            vector = np.asarray(vector, dtype=np.float32)
            norm = np.linalg.norm(vector)
            by_label[label] = vector / norm if norm > 0 else vector

        turns = []
        for segment, _track, speaker_label in diarization.itertracks(yield_label=True):
            embedding = by_label.get(speaker_label)
            if embedding is None:
                continue
            turns.append(SpeakerTurn(
                start=float(segment.start), end=float(segment.end),
                speaker_label=speaker_label, embedding=embedding,
            ))
        return turns
