"""Entity extraction + MENTIONS edge creation over every text-bearing node."""
from __future__ import annotations

import logging

from enrichment.models.text import TextEncoder

from .config import GraphSettings
from .extraction import EntityExtractor, entities_from_detections, normalize_name
from .repository import GraphRepository

log = logging.getLogger(__name__)


def build_entities_and_mentions(
    repository: GraphRepository,
    extractor: EntityExtractor,
    text_encoder: TextEncoder,
    case_id: str,
    settings: GraphSettings,
    only_pending: bool = True,
) -> tuple[int, int]:
    """Returns (entities_touched, mentions_created)."""
    nodes = repository.fetch_text_nodes(case_id, only_pending=only_pending)
    log.info("extracting entities from %d text-bearing node(s)", len(nodes))

    entities_seen: set[str] = set()
    mentions_created = 0

    for index, node in enumerate(nodes, start=1):
        if index % 25 == 0:
            log.info("entity extraction: %d/%d nodes processed", index, len(nodes))

        found = []
        if settings.enable_entity_extraction and extractor.available:
            found.extend(extractor.extract(node.text_content))
        found.extend(entities_from_detections(node.metadata))

        for candidate in found:
            normalized = normalize_name(candidate.name)
            if not normalized:
                continue

            embedding = text_encoder.embed(candidate.name)
            entity_id = repository.upsert_entity(
                case_id, candidate.entity_type, candidate.name, normalized, embedding
            )
            entities_seen.add(entity_id)
            if repository.add_mention(entity_id, node.id, candidate.name, candidate.source):
                mentions_created += 1

    repository.commit()
    log.info(
        "entity extraction: %d unique entit(ies), %d mention(s) created",
        len(entities_seen), mentions_created,
    )
    return len(entities_seen), mentions_created
