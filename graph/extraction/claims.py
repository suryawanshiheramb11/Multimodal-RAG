"""Claim extraction and contradiction judging via the local LLM.

Two separate prompts, deliberately: distilling one node's text into a claim is
a per-node cost paid once and cached on the node, while judging a *pair* is
paid per candidate pair. Fusing them into one prompt would re-derive both
claims for every pair a node participates in.

Both classes reuse enrichment's Captioner (the ollama HTTP wrapper) for the
same reason `EntityExtractor` does: this phase runs after enrichment and
shouldn't reimplement a collaborator it already has.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from enrichment.models.captioning import Captioner

from ..config import (
    CLAIM_EXTRACTION_PROMPT,
    CONTRADICTION_PROMPT,
    NAME_EXTRACTION_PROMPT,
    RELATION_EDGE_TYPES,
    GraphSettings,
)
from .json_response import parse_json_object

log = logging.getLogger(__name__)

#: The model's escape hatch for text that asserts nothing. Compared
#: case-insensitively against the whole (stripped) response, so a model that
#: answers "None." or "NONE" is understood, while a real claim that merely
#: contains the word "none" is not discarded.
_NO_CLAIM = {"none", "none.", "no claim", "n/a"}

#: Verdicts the judge may return. Anything else is treated as a malformed
#: response rather than silently coerced into an edge.
_VALID_RELATIONS = frozenset({"contradicts", "corroborates", "unrelated"})


@dataclass(frozen=True)
class ClaimVerdict:
    """How two claims relate, as judged by the LLM."""

    relation: str  # 'contradicts' | 'corroborates' | 'unrelated'
    confidence: float
    explanation: str

    @property
    def edge_type(self) -> str | None:
        """The relationship_type to store, or None for 'unrelated'."""
        return RELATION_EDGE_TYPES.get(self.relation)


class ClaimExtractor:
    """Reduces a node's text to one subject-verb-object assertion."""

    def __init__(self, captioner: Captioner, settings: GraphSettings) -> None:
        self._captioner = captioner
        self._settings = settings

    @property
    def available(self) -> bool:
        return self._captioner.available

    @property
    def unavailable_reason(self) -> str | None:
        return self._captioner.unavailable_reason

    def extract(self, text: str) -> str | None:
        """Return the claim, or None when the text asserts nothing."""
        if not text or not text.strip():
            return None

        truncated = text[: self._settings.max_claim_chars]
        response = self._captioner.complete(CLAIM_EXTRACTION_PROMPT.format(text=truncated))
        return self._parse(response)

    @staticmethod
    def _parse(response: str | None) -> str | None:
        if not response:
            return None

        claim = " ".join(response.split()).strip()
        if not claim or claim.lower() in _NO_CLAIM:
            return None
        return claim


class ContradictionJudge:
    """Asks the LLM whether two claims contradict, corroborate, or neither."""

    def __init__(self, captioner: Captioner, settings: GraphSettings) -> None:
        self._captioner = captioner
        self._settings = settings

    @property
    def available(self) -> bool:
        return self._captioner.available

    @property
    def unavailable_reason(self) -> str | None:
        return self._captioner.unavailable_reason

    def compare(self, claim_a: str, claim_b: str) -> ClaimVerdict | None:
        """Return the verdict, or None if the model gave nothing usable."""
        if not claim_a or not claim_b:
            return None

        response = self._captioner.complete(
            CONTRADICTION_PROMPT.format(claim_a=claim_a, claim_b=claim_b), json_mode=True
        )
        return self._parse(response)

    def _parse(self, response: str | None) -> ClaimVerdict | None:
        data = parse_json_object(response)
        if data is None:
            return None

        relation = str(data.get("relation") or "").strip().lower()
        if relation not in _VALID_RELATIONS:
            log.warning("contradiction judge returned unknown relation %r", relation)
            return None

        return ClaimVerdict(
            relation=relation,
            confidence=self._confidence(data.get("confidence")),
            explanation=str(data.get("explanation") or "").strip(),
        )

    def _confidence(self, raw) -> float:
        """The model's own confidence when it gives a usable one.

        Small models return this as a bare number, a percentage, or a word;
        anything that isn't a real number in [0, 1] falls back to the
        configured default rather than storing a nonsense score.
        """
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return self._settings.contradiction_default_confidence
        if not 0.0 <= value <= 1.0:
            return self._settings.contradiction_default_confidence
        return value


class SpeakerNameExtractor:
    """Pulls a speaker's own name out of a transcript snippet, for naming an
    identity that face+voice fusion just created.

    Shares `_NO_CLAIM` with ClaimExtractor: both prompts use the same NONE
    escape hatch, and both need it for the same reason — most transcripts
    never name their speaker, and a model guessing from tone would poison an
    identity's display name with something false.
    """

    def __init__(self, captioner: Captioner, settings: GraphSettings) -> None:
        self._captioner = captioner
        self._settings = settings

    @property
    def available(self) -> bool:
        return self._captioner.available

    @property
    def unavailable_reason(self) -> str | None:
        return self._captioner.unavailable_reason

    def extract(self, transcript: str) -> str | None:
        if not transcript or not transcript.strip():
            return None

        truncated = transcript[: self._settings.max_claim_chars]
        response = self._captioner.complete(NAME_EXTRACTION_PROMPT.format(text=truncated))
        return self._parse(response)

    @staticmethod
    def _parse(response: str | None) -> str | None:
        if not response:
            return None
        name = " ".join(response.split()).strip().strip("\"'")
        if not name or name.lower() in _NO_CLAIM:
            return None
        # A real name is short; a model that ignored the "respond with ONLY
        # the name" instruction and answered in a sentence is not usable.
        if len(name) > 60 or len(name.split()) > 5:
            return None
        return name
