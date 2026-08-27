"""Natural-language question answering over the evidence graph.

Deliberately not "ask an LLM and hope for the best." A question goes through
three stages, and only the last one ever calls a model:

  1. classify  — plain keyword/regex matching decides which of a handful of
                 fixed intents the question is (no model call, pure function)
  2. retrieve  — ordinary SQL against the graph phases 3-7 already built
                 (no model call — see the "question answering" section of
                 repository.py)
  3. synthesize — the LLM turns an already-complete, already-correct list of
                  facts into one readable sentence. It cannot invent a fact
                  that isn't in the list it was handed, and if there is no
                  model available, or nothing to synthesize, the answer falls
                  back to a template built with no model call at all.

This trades recall for honesty: a question phrased in a way the keyword
matcher and SQL patterns don't anticipate gets a shrug or a best-effort
semantic-search fallback, not a fluent-sounding guess.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from enrichment.models.captioning import Captioner
from enrichment.models.text import TextEncoder

from .config import ANSWER_SYNTHESIS_PROMPT, GraphSettings
from .repository import GraphRepository

log = logging.getLogger(__name__)


# -- stage 1: classification (pure — no I/O, no model call) -------------------

@dataclass(frozen=True)
class QuestionIntent:
    kind: str  # 'contradiction' | 'timeline' | 'identity' | 'co_occurrence' | 'entity' | 'general'
    subject: str | None
    #: Only set for 'timeline': True means "look before this point in time",
    #: False means "after", None means "around" (a window centred on it).
    before: bool | None = None


_CONTRADICTION_WORDS = re.compile(r"\b(contradict|disagree|conflict|corroborat|dispute)\w*", re.I)
_CO_OCCURRENCE_PATTERN = re.compile(
    r"\b(who (?:was|is|were)?\s*(?:present|there|with|near)|who else|co-?occur)", re.I
)
_IDENTITY_PATTERN = re.compile(r"\b(who is|who'?s|identity of|which person)\b", re.I)
_BEFORE_PATTERN = re.compile(r"\bbefore\b", re.I)
_AFTER_PATTERN = re.compile(r"\bafter\b", re.I)
_TIMELINE_PATTERN = re.compile(
    r"\b(when did|what happened|timeline|before|after|around|during)\b", re.I
)
_TIME_TOKEN = re.compile(r"\b\d{1,2}:\d{2}\b|\b\d{1,4}(?:\.\d+)?\s*(?:sec|second)s?\b", re.I)

#: Stripped out before what's left is treated as "the subject" — question
#: words, connectors, and the intent-trigger words themselves (leaving those
#: in would make "who is present" extract "present" as the subject of a
#: co-occurrence question about nothing in particular).
_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "did", "does", "do", "who",
    "what", "when", "where", "why", "how", "any", "there", "about", "regarding",
    "involving", "on", "in", "of", "to", "we", "have", "has", "evidence", "at",
    "tell", "me", "show", "please", "and", "or", "with", "present", "else",
    "before", "after", "around", "during", "happened", "contradict",
    "contradicts", "contradiction", "contradictions", "disagree", "disagrees",
    "disagreement", "conflict", "conflicts", "corroborate", "corroborates",
    "identity", "person", "this", "that", "which", "s", "mentioned",
    "mentions", "mention", "involved", "involves",
}


def _extract_subject(question: str) -> str | None:
    """A best-effort guess at "the thing this question is about" — a keyword
    filter, not NLP, which is the point: nothing here calls a model.

    An explicit "about X" / "regarding X" / "involving X" clause is trusted
    over the whole sentence when present, since whatever follows it is almost
    always the real subject.
    """
    clause = re.search(r"\b(?:about|regarding|involving|of)\s+(.+?)[?.!]*$", question, re.I)
    text = clause.group(1) if clause else question
    words = re.findall(r"[A-Za-z][A-Za-z'-]*", text)
    kept = [w for w in words if w.lower() not in _STOPWORDS]
    subject = " ".join(kept).strip()
    return subject or None


_KEYWORD_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9'-]*")


def _extract_keywords(question: str, max_terms: int = 6) -> list[str]:
    """Distinctive words worth a literal substring match, alongside the
    semantic search in `_retrieve_general`.

    A whole-question embedding is a poor proxy for an exact code or ID: "what
    is the PNR number" does not land, in embedding space, near a page whose
    only relevant content is a six-character alphanumeric string, so cosine
    similarity can — and in practice does — rank the one node that has it
    below several that don't. A plain substring match doesn't share that blind
    spot, so long as it isn't fed near-universal words that would match almost
    every node; the stopword list and a floor on word length keep it to terms
    actually worth matching on.
    """
    seen: list[str] = []
    for word in _KEYWORD_TOKEN.findall(question):
        lower = word.lower()
        if lower in _STOPWORDS or len(word) < 3 or lower in seen:
            continue
        seen.append(lower)
        if len(seen) >= max_terms:
            break
    return seen


def classify_question(question: str) -> QuestionIntent:
    q = question.strip()

    if _CONTRADICTION_WORDS.search(q):
        return QuestionIntent("contradiction", _extract_subject(q))
    if _CO_OCCURRENCE_PATTERN.search(q):
        return QuestionIntent("co_occurrence", _extract_subject(q))
    if _IDENTITY_PATTERN.search(q):
        return QuestionIntent("identity", _extract_subject(q))
    if _TIMELINE_PATTERN.search(q) or _TIME_TOKEN.search(q):
        if _BEFORE_PATTERN.search(q):
            before = True
        elif _AFTER_PATTERN.search(q):
            before = False
        else:
            before = None
        return QuestionIntent("timeline", _extract_subject(q), before=before)

    subject = _extract_subject(q)
    return QuestionIntent("entity", subject) if subject else QuestionIntent("general", None)


def _parse_seconds(question: str) -> float | None:
    """A literal time reference in the question, if there is one — "12:03"
    or "125 seconds" — resolved before falling back to an entity's own
    time range for "before X" / "after X" questions."""
    match = re.search(r"\b(\d{1,2}):(\d{2})\b", question)
    if match:
        return int(match.group(1)) * 60 + int(match.group(2))
    match = re.search(r"\b(\d{1,4}(?:\.\d+)?)\s*(?:sec|second)s?\b", question, re.I)
    return float(match.group(1)) if match else None


# -- facts: the shape retrieval hands to both the prompt and the fallback ----

@dataclass(frozen=True)
class Fact:
    label: str
    detail: str = ""
    node_id: str | None = None

    def line(self) -> str:
        return f"- {self.label}: {self.detail}" if self.detail else f"- {self.label}"


@dataclass
class Answer:
    question: str
    intent: str
    text: str
    facts: list[Fact] = field(default_factory=list)
    source_node_ids: list[str] = field(default_factory=list)
    used_llm: bool = False


def _where(row: dict) -> str:
    if row.get("start_time") is not None:
        return f"{row['start_time']:.1f}-{row.get('end_time') or row['start_time']:.1f}s"
    if row.get("page_number") is not None:
        return f"p{row['page_number']}"
    return ""

def _content(row: dict) -> str:
    """The text shown for a fact — and, through synthesis, all the LLM ever
    sees of this node.

    `claim` (when present) is a single sentence a phase-5 LLM call reduced the
    node's full text down to for pairwise contradiction comparison — a lossy
    summary by design. Preferring it here for every intent, not just
    contradiction, meant a question could retrieve exactly the right node and
    still get "no evidence of that": a ticket page's `claim` might be
    "Ticketed on 05 Aug 2025 21:58" while the PNR the question asked about
    sat further down the page's actual text, present in `text_content` but
    never reaching the synthesizer. `text_content` is the source `claim` was
    derived from, so it is always at least as complete; `claim` is only used
    when a node genuinely has no text_content of its own.
    """
    text = row.get("text_content") or row.get("claim") or ""
    text = " ".join(text.split())
    return text if len(text) <= 220 else text[:219].rstrip() + "…"


def _node_fact(row: dict) -> Fact:
    where = _where(row)
    label = f"{row['node_type']} " + (f"at {where} " if where else "") + f"in {row['file_name']}"
    return Fact(label=label.strip(), detail=_content(row), node_id=str(row["id"]))


# -- stage 2: retrieval (SQL only — no model call) ----------------------------

def _retrieve_entity(
    repository: GraphRepository, case_id: str, subject: str | None, limit: int
) -> list[Fact]:
    if not subject:
        return []
    matches = repository.entities_mentioning_text(case_id, subject)
    if not matches:
        return []
    node_ids = [n for m in matches for n in m["node_ids"]][:limit]
    rows = repository.fetch_nodes_by_ids(node_ids)
    return [_node_fact(row) for row in rows]


def _retrieve_contradiction(
    repository: GraphRepository, case_id: str, subject: str | None, limit: int
) -> list[Fact]:
    relations = repository.fetch_relations_about(
        case_id, subject, ["CONTRADICTS", "CORROBORATES"], limit=limit
    )
    facts = []
    for r in relations:
        label = (
            f"{r['relationship_type']} between {r['subject_node_type']} in "
            f"{r['subject_file']} and {r['object_node_type']} in {r['object_file']}"
        )
        facts.append(
            Fact(label=label, detail=r["explanation"] or "", node_id=str(r["subject_node_id"]))
        )
    return facts


def _retrieve_identity(
    repository: GraphRepository, case_id: str, subject: str | None, limit: int
) -> list[Fact]:
    if not subject:
        return []
    identities = repository.fetch_identities_matching(case_id, subject)
    if not identities:
        return []
    facts = []
    for identity in identities[:3]:  # a name match is rarely ambiguous across more than a couple
        evidence = repository.fetch_identity_evidence(case_id, str(identity["id"]))
        name = identity["display_name"] or "unnamed identity"
        for row in evidence[:limit]:
            where = f" at {row['start_time']:.1f}s" if row.get("start_time") is not None else ""
            label = f"{name}: {row['node_type']}{where} in {row['file_name']} (via {row['via']})"
            facts.append(Fact(label=label, detail=_content(row), node_id=str(row["node_id"])))
    return facts


def _retrieve_co_occurrence(
    repository: GraphRepository, case_id: str, subject: str | None, limit: int
) -> list[Fact]:
    if not subject:
        return []
    matches = repository.entities_mentioning_text(case_id, subject)
    node_ids = [n for m in matches for n in m["node_ids"]][:limit]

    facts = [
        Fact(
            label=f"{other['canonical_name']} ({other['entity_type']})",
            detail=f"co-mentioned in {other['shared_nodes']} shared node(s)",
            node_id=other["sample_node_id"],
        )
        for other in repository.fetch_co_mentioned_entities(case_id, subject, limit=limit)
    ]
    facts.extend(
        Fact(
            label=(identity["display_name"] or "unnamed identity"),
            detail=f"linked to this evidence via {identity['via']}",
            node_id=None,
        )
        for identity in repository.fetch_identities_for_nodes(case_id, node_ids)
    )
    return facts


def _retrieve_timeline(
    repository: GraphRepository, case_id: str, intent: QuestionIntent, question: str,
    settings: GraphSettings, limit: int,
) -> list[Fact]:
    anchor = _parse_seconds(question)
    if anchor is None and intent.subject:
        bounds = repository.fetch_entity_time_bounds(case_id, intent.subject)
        if bounds:
            anchor = bounds[0] if intent.before else bounds[1]
    if anchor is None:
        return []

    if intent.before is True:
        window = (None, anchor)
    elif intent.before is False:
        window = (anchor, None)
    else:
        half = settings.timeline_window_sec
        window = (max(0.0, anchor - half), anchor + half)

    pack = repository.fetch_evidence_pack(case_id, start_time=window[0], end_time=window[1])
    facts = []
    for node in pack[:limit]:
        where = f"{node['start_time']:.1f}s" if node.get("start_time") is not None else "?"
        label = f"{node['node_type']} at {where} in {node['file_name']}"
        facts.append(Fact(label=label, detail=_content(node), node_id=node["node_id"]))
    return facts


def _retrieve_general(
    repository: GraphRepository, text_encoder: TextEncoder, case_id: str, question: str, limit: int,
) -> list[Fact]:
    """Semantic search, backstopped by a literal keyword match.

    Run both and merge rather than picking one: they fail in opposite ways.
    Semantic search finds paraphrase and context but can bury an exact code
    or name under more "topically similar" nodes; keyword match finds the
    exact string but knows nothing about meaning. A question with a
    distinctive term in it (an ID, a name, an acronym) gets that node
    regardless of where cosine similarity happened to rank it.
    """
    keyword_facts: list[Fact] = []
    keywords = _extract_keywords(question)
    if keywords:
        rows = repository.search_nodes_by_keyword(case_id, keywords, limit=limit)
        keyword_facts = [_node_fact(row) for row in rows]

    semantic_facts: list[Fact] = []
    if text_encoder.available:
        vector = text_encoder.embed(question)
        if vector is not None:
            rows = repository.search_nodes_by_text(case_id, vector, limit=limit)
            semantic_facts = [
                _node_fact(row) for row in rows
                if row["score"] is not None and row["score"] >= 0.2
            ]

    seen_ids = {f.node_id for f in keyword_facts}
    combined = keyword_facts + [f for f in semantic_facts if f.node_id not in seen_ids]
    return combined[:limit]


def retrieve_facts(
    repository: GraphRepository, text_encoder: TextEncoder, case_id: str,
    intent: QuestionIntent, question: str, settings: GraphSettings,
) -> list[Fact]:
    limit = settings.max_qa_facts
    if intent.kind == "entity":
        facts = _retrieve_entity(repository, case_id, intent.subject, limit)
    elif intent.kind == "contradiction":
        facts = _retrieve_contradiction(repository, case_id, intent.subject, limit)
    elif intent.kind == "identity":
        facts = _retrieve_identity(repository, case_id, intent.subject, limit)
    elif intent.kind == "co_occurrence":
        facts = _retrieve_co_occurrence(repository, case_id, intent.subject, limit)
    elif intent.kind == "timeline":
        facts = _retrieve_timeline(repository, case_id, intent, question, settings, limit)
    else:
        facts = []

    # A specific intent that came up empty still gets one more chance through
    # semantic search rather than reporting "no evidence" outright — the
    # keyword classifier can be right about *that* a question is about X
    # while still missing evidence phrased differently from X.
    if not facts:
        facts = _retrieve_general(repository, text_encoder, case_id, question, limit)
    return facts[:limit]


# -- stage 3: synthesis (the only stage that may call a model) ---------------

_INTENT_LEAD = {
    "contradiction": "Disagreements found",
    "co_occurrence": "Co-occurring evidence found",
    "identity": "Evidence linked to this person",
    "timeline": "Evidence in that time range",
    "entity": "Evidence mentioning this",
    "general": "Closest matching evidence",
}


def _template_answer(intent: QuestionIntent, facts: list[Fact]) -> str:
    """A deterministic answer built with no model call at all — the fallback
    when synthesis is disabled, the model is unavailable, or it declined to
    produce usable text. Less readable than a synthesized sentence, but never
    unavailable and never wrong about what it lists."""
    lead = _INTENT_LEAD.get(intent.kind, "Evidence found")
    lines = "\n".join(fact.line() for fact in facts)
    return f"{lead} ({len(facts)} item(s)):\n{lines}"


def _no_facts_answer(intent: QuestionIntent) -> str:
    return "No evidence in this case matches that question."


def synthesize_answer(
    captioner: Captioner | None, question: str, intent: QuestionIntent,
    facts: list[Fact], settings: GraphSettings,
) -> tuple[str, bool]:
    """Returns (answer_text, used_llm)."""
    if not facts:
        return _no_facts_answer(intent), False

    if settings.enable_qa_llm_synthesis and captioner is not None and captioner.available:
        prompt = ANSWER_SYNTHESIS_PROMPT.format(
            question=question, facts="\n".join(f.line() for f in facts)
        )
        response = captioner.complete(prompt)
        if response and response.strip():
            return response.strip(), True

    return _template_answer(intent, facts), False


# -- orchestration -------------------------------------------------------------

def answer_question(
    repository: GraphRepository, text_encoder: TextEncoder, captioner: Captioner | None,
    case_id: str, question: str, settings: GraphSettings | None = None,
) -> Answer:
    settings = settings or GraphSettings()
    question = question.strip()
    if not question:
        return Answer(question=question, intent="general", text="Ask a question about the case.")

    intent = classify_question(question)
    facts = retrieve_facts(repository, text_encoder, case_id, intent, question, settings)
    text, used_llm = synthesize_answer(captioner, question, intent, facts, settings)

    log.info(
        "qa: intent=%s facts=%d llm=%s question=%r",
        intent.kind, len(facts), used_llm, question[:80],
    )
    return Answer(
        question=question, intent=intent.kind, text=text, facts=facts,
        source_node_ids=[f.node_id for f in facts if f.node_id], used_llm=used_llm,
    )
