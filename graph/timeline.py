"""Timeline event grouping: cluster evidence nodes that occur close together
in time and either share an entity or read as semantically similar, then
summarize each group into one timeline_event row with SAME_EVENT edges.

Grouping (union-find over time+similarity) uses only node timestamps taken
at face value. Nodes from different source files use their own file-relative
clock — genuinely aligning those clocks is exactly the offset-estimation
problem `crossmodal.verify_audio_alignment_prep` is prep for, and is left to
a later phase. Until then, grouping is most meaningful within one file and
across files that already share a clock (e.g. synced multi-camera footage).
"""
from __future__ import annotations

import logging

import numpy as np

from enrichment.models.captioning import Captioner

from .config import GraphSettings, TIMELINE_EVENT_PROMPT
from .repository import EventCandidate, GraphRepository

log = logging.getLogger(__name__)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


class _UnionFind:
    def __init__(self, keys: list[str]) -> None:
        self._parent = {k: k for k in keys}

    def find(self, key: str) -> str:
        while self._parent[key] != key:
            self._parent[key] = self._parent[self._parent[key]]
            key = self._parent[key]
        return key

    def union(self, a: str, b: str) -> None:
        root_a, root_b = self.find(a), self.find(b)
        if root_a != root_b:
            self._parent[root_a] = root_b

    def groups(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for key in self._parent:
            out.setdefault(self.find(key), []).append(key)
        return out


def group_into_events(
    candidates: list[EventCandidate],
    entities_by_node: dict[str, set[str]],
    settings: GraphSettings,
) -> list[list[EventCandidate]]:
    """Connected components over "within the time window AND (shares an
    entity OR high text similarity)". Pure function: no I/O, easy to test
    against synthetic candidates.

    A single-node "group" is not an event worth recording, so those are
    dropped — the spec's grouping mechanism is inherently about correlating
    two or more pieces of evidence.
    """
    by_id = {c.id: c for c in candidates}
    ordered = sorted(candidates, key=lambda c: c.start_time)
    uf = _UnionFind([c.id for c in ordered])

    for i, a in enumerate(ordered):
        for b in ordered[i + 1:]:
            if b.start_time - a.start_time > settings.timeline_window_sec:
                break  # sorted by start_time: nothing further is in range either

            shares_entity = bool(
                entities_by_node.get(a.id, set()) & entities_by_node.get(b.id, set())
            )
            similar_text = (
                a.text_embedding is not None and b.text_embedding is not None
                and _cosine(a.text_embedding, b.text_embedding) > settings.timeline_text_similarity_threshold
            )
            if shares_entity or similar_text:
                uf.union(a.id, b.id)

    groups = [
        [by_id[node_id] for node_id in member_ids]
        for member_ids in uf.groups().values()
        if len(member_ids) >= 2
    ]
    groups.sort(key=lambda group: min(c.start_time for c in group))
    return groups


def _describe_group(
    captioner: Captioner | None, group: list[EventCandidate], settings: GraphSettings
) -> str:
    """LLM summary when available, a plain templated fallback otherwise — a
    missing/unreachable model must not stop timeline events from being
    recorded, only from being nicely worded."""
    start = min(c.start_time for c in group)
    end = max(c.end_time for c in group)
    fallback = f"{len(group)} evidence node(s) between {start:.1f}s and {end:.1f}s"

    if not settings.enable_timeline_llm_summary or captioner is None or not captioner.available:
        return fallback

    snippets = "\n".join(
        f"- ({c.node_type}) {c.text_content.strip()[:300]}" for c in group if c.text_content
    )
    if not snippets:
        return fallback

    description = captioner.complete(TIMELINE_EVENT_PROMPT.format(snippets=snippets))
    return description.strip() if description else fallback


def build_timeline_events(
    repository: GraphRepository, captioner: Captioner | None, case_id: str, settings: GraphSettings
) -> tuple[int, int]:
    """Returns (events_created, same_event_links_created)."""
    repository.clear_timeline_events(case_id)  # idempotent: regroup, don't accumulate

    candidates = repository.fetch_event_candidates(case_id)
    entities_by_node = repository.fetch_entities_by_node(case_id)
    groups = group_into_events(candidates, entities_by_node, settings)
    log.info("timeline events: %d candidate node(s) -> %d group(s)", len(candidates), len(groups))

    events_created = 0
    links_created = 0
    for group in groups:
        description = _describe_group(captioner, group, settings)
        event_id = repository.insert_timeline_event(
            case_id, description,
            start_time=min(c.start_time for c in group),
            end_time=max(c.end_time for c in group),
            node_ids=[c.id for c in group],
        )
        events_created += 1
        for candidate in group:
            if repository.link_node_to_event(event_id, candidate.id):
                links_created += 1

    repository.commit()
    log.info(
        "timeline events: %d event(s) created, %d SAME_EVENT edge(s)",
        events_created, links_created,
    )
    return events_created, links_created
