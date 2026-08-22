"""Phase 5: contradiction and corroboration detection.

Three stages, each isolated so the expensive one is entered only with work
that has already been justified:

  1. claim extraction  — one LLM call per node, cached on the node
  2. candidate pairing — pure set logic over edges already in the graph
  3. judging           — one LLM call per surviving pair

Stage 2 is what keeps this affordable. Comparing every node against every
other is O(n^2) LLM calls, which is minutes of local inference per hundred
nodes; instead only pairs the graph *already* connected are considered — two
nodes mentioning the same entity, two nodes on an existing cross-modal edge,
or two nodes grouped into the same timeline event — and those are then
filtered again by text-embedding similarity before any call is made.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

from .config import GraphSettings
from .extraction.claims import ClaimExtractor, ContradictionJudge
from .repository import ClaimRecord, GraphRepository

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class CandidatePair:
    """Two nodes worth comparing, and why they were proposed.

    `origins` is kept (rather than discarded once the pair is formed) because
    it is stored on the resulting edge: a reviewer asking "why were these two
    ever compared?" gets an answer from the row itself.
    """

    node_a_id: str
    node_b_id: str
    origins: frozenset[str]


@dataclass
class ContradictionReport:
    claims_extracted: int = 0
    pairs_proposed: int = 0
    pairs_judged: int = 0
    contradicts: int = 0
    corroborates: int = 0
    unrelated: int = 0
    skipped: dict[str, int] = field(default_factory=dict)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denominator) if denominator > 0 else 0.0


# -- stage 1: claims ---------------------------------------------------------


def extract_claims(
    repository: GraphRepository,
    extractor: ClaimExtractor,
    case_id: str,
    settings: GraphSettings,
) -> int:
    """Distil each pending node's text into one claim. Returns how many nodes
    produced one.

    A node the model finds no claim in is still marked as attempted, so the
    next run doesn't pay for it again.
    """
    if not extractor.available:
        log.warning("claim extraction skipped: %s", extractor.unavailable_reason)
        return 0

    nodes = repository.fetch_claim_candidates(case_id, settings.claim_node_types)
    log.info("extracting claims from %d node(s)", len(nodes))

    extracted = 0
    for index, node in enumerate(nodes, start=1):
        if index % 25 == 0:
            log.info("claim extraction: %d/%d nodes processed", index, len(nodes))

        claim = extractor.extract(node.text_content)
        repository.store_claim(node.id, claim)
        if claim:
            extracted += 1

    repository.commit()
    log.info("claim extraction: %d/%d node(s) yielded a claim", extracted, len(nodes))
    return extracted


# -- stage 2: candidate pairs ------------------------------------------------


def build_candidate_pairs(
    entity_groups: list[tuple[str, str, list[str]]],
    alignment_pairs: list[tuple[str, str, str]],
    event_groups: list[tuple[str, list[str]]],
    settings: GraphSettings,
) -> list[CandidatePair]:
    """Merge the three candidate sources into one deduplicated pair list.

    Pure function — no I/O — so the pairing rules can be tested directly.
    A pair reached by several routes (entity *and* SAME_EVENT, say) is one
    pair carrying both origins, not two comparisons of the same two nodes.
    """
    origins: dict[tuple[str, str], set[str]] = {}

    def add(node_a: str, node_b: str, origin: str) -> None:
        if node_a == node_b:
            return
        key = (min(node_a, node_b), max(node_a, node_b))
        origins.setdefault(key, set()).add(origin)

    for _entity_id, name, node_ids in entity_groups:
        # A very common entity is capped rather than dropped: the first N
        # nodes still get compared, which is more useful than skipping the
        # entity entirely and never noticing a conflict about it.
        capped = sorted(node_ids)[: settings.max_nodes_per_entity_for_pairs]
        if len(capped) < len(node_ids):
            log.info(
                "entity %r mentioned by %d nodes; capped to %d for pairing",
                name, len(node_ids), len(capped),
            )
        for node_a, node_b in combinations(capped, 2):
            add(node_a, node_b, f"entity:{name}")

    for node_a, node_b, alignment_type in alignment_pairs:
        add(node_a, node_b, alignment_type)

    for _event_id, node_ids in event_groups:
        for node_a, node_b in combinations(sorted(node_ids), 2):
            add(node_a, node_b, "SAME_EVENT")

    return [
        CandidatePair(node_a_id=a, node_b_id=b, origins=frozenset(reasons))
        for (a, b), reasons in sorted(origins.items())
    ]


def filter_pairs_by_similarity(
    pairs: list[CandidatePair],
    claims: dict[str, ClaimRecord],
    settings: GraphSettings,
) -> tuple[list[CandidatePair], dict[str, int]]:
    """Drop pairs that cannot be judged, or that are too dissimilar to bother.

    Returns the survivors and a tally of why the rest were dropped. A pair
    where either side has no text embedding is *kept*: the pre-filter is an
    optimisation, and a missing vector is not evidence that two claims agree.
    """
    survivors: list[CandidatePair] = []
    skipped = {"no_claim": 0, "below_similarity": 0}

    for pair in pairs:
        claim_a = claims.get(pair.node_a_id)
        claim_b = claims.get(pair.node_b_id)
        if claim_a is None or claim_b is None:
            skipped["no_claim"] += 1
            continue

        if claim_a.text_embedding is not None and claim_b.text_embedding is not None:
            similarity = _cosine(claim_a.text_embedding, claim_b.text_embedding)
            if similarity < settings.contradiction_similarity_threshold:
                skipped["below_similarity"] += 1
                continue

        survivors.append(pair)

    return survivors, skipped


# -- stage 3: judging --------------------------------------------------------


def detect_contradictions(
    repository: GraphRepository,
    judge: ContradictionJudge,
    case_id: str,
    settings: GraphSettings,
) -> ContradictionReport:
    """Judge every surviving candidate pair and store the verdicts as edges."""
    report = ContradictionReport()

    if not judge.available:
        log.warning("contradiction detection skipped: %s", judge.unavailable_reason)
        return report

    repository.clear_claim_relationships(case_id)  # idempotent: re-judge, don't accumulate

    claims = repository.fetch_claims(case_id)
    if not claims:
        log.info("contradiction detection: no claims stored; nothing to compare")
        return report

    pairs = build_candidate_pairs(
        repository.fetch_entity_node_groups(case_id),
        repository.fetch_alignment_pairs(case_id, settings.contradiction_alignment_types),
        repository.fetch_event_node_groups(case_id),
        settings,
    )
    report.pairs_proposed = len(pairs)

    judgeable, report.skipped = filter_pairs_by_similarity(pairs, claims, settings)
    log.info(
        "contradiction detection: %d pair(s) proposed, %d to judge "
        "(%d without a claim, %d below the similarity floor)",
        len(pairs), len(judgeable), report.skipped["no_claim"],
        report.skipped["below_similarity"],
    )

    if len(judgeable) > settings.max_contradiction_pairs:
        raise ValueError(
            f"{len(judgeable)} pairs to judge exceeds the "
            f"{settings.max_contradiction_pairs}-pair cap; raise "
            "max_contradiction_pairs, raise contradiction_similarity_threshold, "
            "or restrict the case"
        )

    for index, pair in enumerate(judgeable, start=1):
        if index % 10 == 0:
            log.info("contradiction detection: %d/%d pairs judged", index, len(judgeable))

        claim_a = claims[pair.node_a_id]
        claim_b = claims[pair.node_b_id]
        verdict = judge.compare(claim_a.claim, claim_b.claim)
        if verdict is None:
            continue

        report.pairs_judged += 1
        if verdict.edge_type is None:
            report.unrelated += 1
            continue

        created = repository.insert_claim_relationship(
            case_id, pair.node_a_id, pair.node_b_id, verdict.edge_type,
            confidence=verdict.confidence, explanation=verdict.explanation,
            metadata={
                "origins": sorted(pair.origins),
                "claim_a": claim_a.claim,
                "claim_b": claim_b.claim,
            },
        )
        if created:
            if verdict.edge_type == "CONTRADICTS":
                report.contradicts += 1
            else:
                report.corroborates += 1

    repository.commit()
    log.info(
        "contradiction detection: %d CONTRADICTS, %d CORROBORATES, %d unrelated",
        report.contradicts, report.corroborates, report.unrelated,
    )
    return report
