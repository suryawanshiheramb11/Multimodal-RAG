"""Semantic search over the evidence graph.

Two independent vector spaces are already populated by the enrichment phase,
and they answer different questions:

  clip_embedding (512-d)  what a frame *looks like*. CLIP puts images and text
                          in one space, so a typed phrase can be compared
                          directly against a video frame — this is what makes
                          "mountains" find mountain footage in a video nobody
                          captioned as such.
  text_embedding (384-d)  what was *said or written* — MiniLM over the fused
                          transcript + caption + OCR + document text.

Hybrid search runs both and merges, because a query like "someone talking about
the lake" has a visual half and a spoken half that neither space answers alone.

Both columns carry an HNSW cosine index, so ranking happens in Postgres and the
vectors never cross the wire.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

from psycopg2.extras import RealDictCursor

from enrichment.config import EnrichmentSettings
from enrichment.registry import ModelRegistry

log = logging.getLogger(__name__)

SearchMode = Literal["hybrid", "visual", "text"]

#: Cosine floors, per space. CLIP text-to-image similarity is compressed into a
#: much narrower band than text-to-text (a strong visual match scores ~0.30,
#: not ~0.9), so the two spaces need different floors or the visual half is
#: either drowned out or filtered away entirely.
#:
#: For CLIP ViT-B/32 a genuinely matching text/image pair lands around
#: 0.25-0.35 while an unrelated pair sits near 0.15-0.23, so 0.24 is the
#: boundary between the two populations rather than a number tuned to any one
#: library. Raising it trades recall for precision: a correct but unusual
#: match ("a dog" against a photo of a distant dog) can fall below it.
_MIN_SCORE = {"visual": 0.24, "text": 0.25}

#: An absolute floor alone is not enough for CLIP: *every* image scores
#: something against *every* phrase, so a query with no real match still
#: returns the library sorted by noise. Results must also stay within this
#: fraction of the best hit, which is what makes "mountains" return nothing
#: when there are no mountains rather than four confident-looking non-answers.
#: Text similarity spreads far wider, so its gate is gentler — a genuinely
#: relevant second hit can legitimately score half the top one.
_RELATIVE_GATE = {"visual": 0.85, "text": 0.60}

#: How much a hybrid hit's visual score is worth relative to its text score.
#: Text similarity is the more literal signal when it exists, so it leads;
#: the visual score still promotes a frame whose *content* matches a query
#: that its transcript never mentions.
_HYBRID_WEIGHT = {"visual": 0.45, "text": 0.55}


@dataclass(frozen=True)
class SearchHit:
    node_id: str
    source_file_id: str
    file_name: str
    file_type: str
    case_id: str
    node_type: str
    start_time: float | None
    end_time: float | None
    page_number: int | None
    text_content: str | None
    score: float
    #: Which space produced the score, for the "why did this match?" badge.
    matched_on: list[str]

    def to_json(self) -> dict:
        return {
            "node_id": self.node_id,
            "source_file_id": self.source_file_id,
            "file_name": self.file_name,
            "file_type": self.file_type,
            "case_id": self.case_id,
            "node_type": self.node_type,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "page_number": self.page_number,
            "snippet": _snippet(self.text_content),
            "score": round(self.score, 4),
            "matched_on": self.matched_on,
        }


def _snippet(text: str | None, limit: int = 240) -> str | None:
    """Trim fused text down to something a result card can show.

    Enrichment stores text under "Transcript:"/"Visual description:" headings;
    those are useful in a detail view but pure noise in a one-line snippet.
    """
    if not text:
        return None
    cleaned = " ".join(text.split())
    return cleaned if len(cleaned) <= limit else cleaned[: limit - 1].rstrip() + "…"


class SemanticSearch:
    """Ranks evidence nodes against a natural-language query.

    Owns one ModelRegistry so CLIP and MiniLM are loaded once for the process
    rather than per request — a cold CLIP load is seconds, a warm encode is
    milliseconds.
    """

    def __init__(self, settings: EnrichmentSettings | None = None) -> None:
        self._settings = settings or EnrichmentSettings()
        self._models = ModelRegistry.build(self._settings)

    # -- model readiness ----------------------------------------------------

    def warm(self) -> dict[str, str]:
        """Force-load just the two encoders search needs.

        Called at startup in a background thread: it turns a multi-second
        stall on the user's first query into a spinner that resolves before
        they finish typing.
        """
        status = {}
        for name, model in (("clip", self._models.clip), ("text", self._models.text_encoder)):
            status[name] = "ready" if model.available else (
                model.unavailable_reason or "unavailable"
            )
        log.info("search encoders: %s", status)
        return status

    @property
    def status(self) -> dict[str, bool]:
        return {
            "visual": self._models.clip.available,
            "text": self._models.text_encoder.available,
        }

    # -- querying -----------------------------------------------------------

    def search(
        self, conn, query: str, *, mode: SearchMode = "hybrid",
        case_id: str | None = None, limit: int = 40,
    ) -> list[SearchHit]:
        query = (query or "").strip()
        if not query:
            return []

        visual = self._visual_hits(conn, query, case_id, limit) if mode in ("hybrid", "visual") else {}
        textual = self._text_hits(conn, query, case_id, limit) if mode in ("hybrid", "text") else {}

        if mode == "visual":
            merged = visual
        elif mode == "text":
            merged = textual
        else:
            merged = self._merge(visual, textual)

        ranked = sorted(merged.values(), key=lambda hit: hit.score, reverse=True)
        return ranked[:limit]

    def _visual_hits(self, conn, query: str, case_id: str | None, limit: int) -> dict[str, SearchHit]:
        encoder = self._models.clip
        if not encoder.available:
            return {}
        vector = encoder.embed_text(query)
        if vector is None:
            return {}
        return self._query(conn, "clip_embedding", vector, case_id, limit, "visual")

    def _text_hits(self, conn, query: str, case_id: str | None, limit: int) -> dict[str, SearchHit]:
        encoder = self._models.text_encoder
        if not encoder.available:
            return {}
        vector = encoder.embed(query)
        if vector is None:
            return {}
        return self._query(conn, "text_embedding", vector, case_id, limit, "text")

    def _query(
        self, conn, column: str, vector, case_id: str | None, limit: int, space: str,
    ) -> dict[str, SearchHit]:
        """One nearest-neighbour scan against a vector column.

        `column` is chosen from a fixed pair by the callers above, never from
        request data — the repo's rule that no identifier is ever interpolated
        from user input still holds.
        """
        if column not in ("clip_embedding", "text_embedding"):
            raise ValueError(f"unsupported vector column: {column}")

        # Over-fetch so the hybrid merge has candidates to combine before the
        # final cut, rather than intersecting two already-truncated lists.
        fetch = min(limit * 3, 200)
        sql = f"""
            SELECT n.id, n.source_file_id, n.node_type, n.start_time, n.end_time,
                   n.page_number, n.text_content,
                   f.file_name, f.file_type, f.case_id,
                   1 - (n.{column} <=> %(vector)s::vector) AS score
            FROM evidence_node n
            JOIN source_file f ON f.id = n.source_file_id
            WHERE n.{column} IS NOT NULL
              AND (%(case_id)s::uuid IS NULL OR f.case_id = %(case_id)s::uuid)
            ORDER BY n.{column} <=> %(vector)s::vector
            LIMIT %(fetch)s
        """  # noqa: S608 - column is validated against a fixed allowlist above

        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, {"vector": vector, "case_id": case_id, "fetch": fetch})
            rows = cur.fetchall()

        scored = [(row, float(row["score"])) for row in rows if row["score"] is not None]
        if not scored:
            return {}

        # Absolute floor first, then the relative gate against the best
        # surviving hit — applied in that order so a library where everything
        # is weakly similar is cut off entirely instead of having its own
        # noise promoted to "the top result".
        floor = _MIN_SCORE[space]
        above_floor = [(row, score) for row, score in scored if score >= floor]
        if not above_floor:
            return {}

        best = max(score for _, score in above_floor)
        gate = max(floor, best * _RELATIVE_GATE[space])

        return {
            str(row["id"]): SearchHit(
                node_id=str(row["id"]),
                source_file_id=str(row["source_file_id"]),
                file_name=row["file_name"],
                file_type=row["file_type"],
                case_id=str(row["case_id"]),
                node_type=row["node_type"],
                start_time=row["start_time"],
                end_time=row["end_time"],
                page_number=row["page_number"],
                text_content=row["text_content"],
                score=score,
                matched_on=[space],
            )
            for row, score in above_floor
            if score >= gate
        }

    @staticmethod
    def _merge(
        visual: dict[str, SearchHit], textual: dict[str, SearchHit]
    ) -> dict[str, SearchHit]:
        """Combine the two spaces into one ranking.

        A node found in both is genuinely stronger evidence than one found in
        either alone, so its weighted scores add rather than one replacing the
        other. A node found in only one keeps just that space's weighted
        share, which is what stops a single-space hit from outranking a
        both-spaces hit on raw magnitude.
        """
        merged: dict[str, SearchHit] = {}
        for space, hits in (("visual", visual), ("text", textual)):
            weight = _HYBRID_WEIGHT[space]
            for node_id, hit in hits.items():
                existing = merged.get(node_id)
                if existing is None:
                    merged[node_id] = SearchHit(
                        **{**hit.__dict__, "score": hit.score * weight}
                    )
                else:
                    merged[node_id] = SearchHit(
                        **{
                            **existing.__dict__,
                            "score": existing.score + hit.score * weight,
                            "matched_on": sorted({*existing.matched_on, space}),
                        }
                    )
        return merged
