"""LLM-based entity extraction from a node's text.

Reuses enrichment's Captioner (ollama HTTP wrapper) for the completion call and
its TextEncoder for the entity-name embedding — this phase runs after
enrichment and shouldn't reimplement collaborators it already has.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from enrichment.models.captioning import Captioner

from ..config import ENTITY_EXTRACTION_PROMPT, ENTITY_TYPES, GraphSettings

log = logging.getLogger(__name__)

#: Matches the first top-level {...} block, tolerating prose or a code fence
#: the model wrapped the JSON in despite being asked not to.
_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


@dataclass(frozen=True)
class ExtractedEntity:
    entity_type: str
    name: str
    #: How this candidate was found — carried through to the mention row
    #: rather than re-derived later from node metadata.
    source: str = "llm_extraction"


def normalize_name(name: str) -> str:
    """Matching key for dedup: collapsed whitespace, lowercased."""
    return " ".join(name.split()).lower()


class EntityExtractor:
    """Extracts (type, name) pairs from free text via a local LLM."""

    def __init__(self, captioner: Captioner, settings: GraphSettings) -> None:
        self._captioner = captioner
        self._settings = settings

    @property
    def available(self) -> bool:
        return self._captioner.available

    @property
    def unavailable_reason(self) -> str | None:
        return self._captioner.unavailable_reason

    def extract(self, text: str) -> list[ExtractedEntity]:
        if not text or not text.strip():
            return []

        truncated = text[: self._settings.max_extraction_chars]
        prompt = ENTITY_EXTRACTION_PROMPT.format(text=truncated)
        response = self._captioner.complete(prompt, json_mode=True)
        if not response:
            log.warning("entity extraction returned no response")
            return []

        return self._parse(response)

    def _parse(self, response: str) -> list[ExtractedEntity]:
        match = _JSON_BLOCK.search(response)
        payload = match.group(0) if match else response
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            log.warning("entity extraction returned unparsable JSON: %.200s", response)
            return []

        raw_entities = data.get("entities") if isinstance(data, dict) else None
        if not isinstance(raw_entities, list):
            return []

        results = []
        for item in raw_entities:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            entity_type = str(item.get("type") or "").strip().lower()
            if not name:
                continue
            if entity_type not in ENTITY_TYPES:
                entity_type = "other"
            results.append(ExtractedEntity(entity_type=entity_type, name=name))
        return results
