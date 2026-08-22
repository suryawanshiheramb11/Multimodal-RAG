"""Graph-construction orchestration.

Runs the five steps in order — entities/mentions, temporal alignment,
similarity edges, face detection, face clustering — each isolated so that one
step failing (e.g. ollama down) does not prevent the others from running.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from enrichment.models.captioning import Captioner
from enrichment.models.clip import ClipEncoder
from enrichment.models.text import TextEncoder

from .alignment import build_temporal_alignments
from .clustering import cluster_faces
from .config import GraphSettings
from .contradictions import detect_contradictions, extract_claims
from .crossmodal import (
    build_document_video_references,
    build_transcript_visual_links,
    verify_audio_alignment_prep,
)
from .entities import build_entities_and_mentions
from .extraction import ClaimExtractor, ContradictionJudge, EntityExtractor
from .faces import detect_faces
from .models.faces import FaceDetector
from .repository import GraphRepository
from .timeline import build_timeline_events

log = logging.getLogger(__name__)


@dataclass
class GraphReport:
    case_id: str
    step_status: dict[str, str] = field(default_factory=dict)
    entities: int = 0
    mentions: int = 0
    aligns_with: int = 0
    similar_to: int = 0
    faces_detected: int = 0
    face_clusters: int = 0
    references: int = 0
    describes: int = 0
    timeline_events: int = 0
    same_event_links: int = 0
    audio_embedding_coverage: dict = field(default_factory=dict)
    claims_extracted: int = 0
    contradicts: int = 0
    corroborates: int = 0
    pairs_judged: int = 0
    summary: dict = field(default_factory=dict)


class GraphPipeline:
    def __init__(
        self,
        repository: GraphRepository,
        entity_extractor: EntityExtractor,
        text_encoder: TextEncoder,
        face_detector: FaceDetector,
        clip_encoder: ClipEncoder,
        captioner: Captioner,
        claim_extractor: ClaimExtractor,
        contradiction_judge: ContradictionJudge,
        settings: GraphSettings,
    ) -> None:
        self._repository = repository
        self._entity_extractor = entity_extractor
        self._text_encoder = text_encoder
        self._face_detector = face_detector
        self._clip_encoder = clip_encoder
        self._captioner = captioner
        self._claim_extractor = claim_extractor
        self._contradiction_judge = contradiction_judge
        self._settings = settings

    def run(self, case_id: str) -> GraphReport:
        report = GraphReport(case_id=case_id)

        if self._begin_step(report, "entities", self._settings.enable_entity_extraction):
            try:
                report.entities, report.mentions = build_entities_and_mentions(
                    self._repository, self._entity_extractor, self._text_encoder,
                    case_id, self._settings,
                )
                report.step_status["entities"] = "ok"
            except Exception as exc:  # noqa: BLE001 - one step must not abort the rest
                self._fail_step(report, "entities", exc)

        if self._begin_step(report, "temporal_alignment", self._settings.enable_temporal_alignment):
            try:
                report.aligns_with = build_temporal_alignments(
                    self._repository, case_id, self._settings
                )
                report.step_status["temporal_alignment"] = "ok"
            except Exception as exc:  # noqa: BLE001
                self._fail_step(report, "temporal_alignment", exc)

        if self._begin_step(report, "similarity", self._settings.enable_similarity_edges):
            try:
                report.similar_to = self._repository.insert_similarity_edges(
                    case_id, self._settings.similarity_threshold,
                    self._settings.max_nodes_for_similarity,
                )
                report.step_status["similarity"] = "ok"
            except Exception as exc:  # noqa: BLE001
                self._fail_step(report, "similarity", exc)

        if self._begin_step(report, "face_detection", self._settings.enable_face_detection):
            try:
                report.faces_detected = detect_faces(
                    self._repository, self._face_detector, case_id, self._settings
                )
                report.step_status["face_detection"] = "ok"
            except Exception as exc:  # noqa: BLE001
                self._fail_step(report, "face_detection", exc)

        cluster_enabled = self._settings.enable_face_clustering and report.faces_detected > 0
        if self._begin_step(report, "face_clustering", cluster_enabled):
            try:
                report.face_clusters = cluster_faces(self._repository, case_id, self._settings)
                report.step_status["face_clustering"] = "ok"
            except Exception as exc:  # noqa: BLE001
                self._fail_step(report, "face_clustering", exc)
        elif not self._settings.enable_face_clustering:
            pass  # already marked skipped by _begin_step
        elif report.faces_detected == 0:
            report.step_status["face_clustering"] = "skipped: no faces detected"

        if self._begin_step(
            report, "document_video_linking", self._settings.enable_document_video_linking
        ):
            try:
                report.references = build_document_video_references(
                    self._repository, case_id, self._settings
                )
                report.step_status["document_video_linking"] = "ok"
            except Exception as exc:  # noqa: BLE001
                self._fail_step(report, "document_video_linking", exc)

        if self._begin_step(
            report, "transcript_visual_linking", self._settings.enable_transcript_visual_linking
        ):
            try:
                report.describes = build_transcript_visual_links(
                    self._repository, self._clip_encoder, case_id, self._settings
                )
                report.step_status["transcript_visual_linking"] = "ok"
            except Exception as exc:  # noqa: BLE001
                self._fail_step(report, "transcript_visual_linking", exc)

        if self._begin_step(report, "audio_alignment_prep", True):
            try:
                report.audio_embedding_coverage = verify_audio_alignment_prep(
                    self._repository, case_id
                )
                report.step_status["audio_alignment_prep"] = "ok"
            except Exception as exc:  # noqa: BLE001
                self._fail_step(report, "audio_alignment_prep", exc)

        if self._begin_step(report, "timeline_events", self._settings.enable_timeline_events):
            try:
                report.timeline_events, report.same_event_links = build_timeline_events(
                    self._repository, self._captioner, case_id, self._settings
                )
                report.step_status["timeline_events"] = "ok"
            except Exception as exc:  # noqa: BLE001
                self._fail_step(report, "timeline_events", exc)

        if self._begin_step(report, "claim_extraction", self._settings.enable_claim_extraction):
            try:
                report.claims_extracted = extract_claims(
                    self._repository, self._claim_extractor, case_id, self._settings
                )
                report.step_status["claim_extraction"] = "ok"
            except Exception as exc:  # noqa: BLE001
                self._fail_step(report, "claim_extraction", exc)

        if self._begin_step(
            report, "contradiction_detection", self._settings.enable_contradiction_detection
        ):
            try:
                contradiction = detect_contradictions(
                    self._repository, self._contradiction_judge, case_id, self._settings
                )
                report.contradicts = contradiction.contradicts
                report.corroborates = contradiction.corroborates
                report.pairs_judged = contradiction.pairs_judged
                report.step_status["contradiction_detection"] = "ok"
            except Exception as exc:  # noqa: BLE001
                self._fail_step(report, "contradiction_detection", exc)

        report.summary = self._repository.graph_summary(case_id)
        return report

    def _begin_step(self, report: GraphReport, name: str, enabled: bool) -> bool:
        if not enabled:
            report.step_status.setdefault(name, "skipped")
            log.info("step %-18s skipped", name)
        return enabled

    def _fail_step(self, report: GraphReport, name: str, exc: Exception) -> None:
        log.exception("step %s failed", name)
        report.step_status[name] = f"failed: {exc}"


def build_graph_pipeline(conn, settings: GraphSettings | None = None) -> GraphPipeline:
    """Composition root for phases 3 through 5."""
    settings = settings or GraphSettings()
    # Entity extraction, timeline-event summarization, claim extraction and
    # contradiction judging all prompt the same local vision-language model
    # with text-only input, which it handles fine, so one Captioner instance
    # (and one ollama connection check) covers them all rather than each step
    # building its own.
    captioner = Captioner(settings.entity_model, settings.ollama_host, settings.ollama_timeout_sec)
    text_encoder = TextEncoder(settings.text_encoder_model, settings.entity_embedding_dim)
    face_detector = FaceDetector(
        settings.face_model_pack, settings.face_embedding_dim, settings.face_detection_confidence
    )
    clip_encoder = ClipEncoder(settings.clip_model, settings.clip_embedding_dim)
    return GraphPipeline(
        repository=GraphRepository(conn),
        entity_extractor=EntityExtractor(captioner, settings),
        text_encoder=text_encoder,
        face_detector=face_detector,
        clip_encoder=clip_encoder,
        captioner=captioner,
        claim_extractor=ClaimExtractor(captioner, settings),
        contradiction_judge=ContradictionJudge(captioner, settings),
        settings=settings,
    )
