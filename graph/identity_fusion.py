"""Phase 7: cross-modal identity fusion.

Face clustering (phase 3) and voice clustering (this phase) each produce
clusters that are only known to be *a* person, not *which* person, and the
two are in unrelated embedding spaces — a face vector and a voice vector for
the same human being are not comparable to each other. What ties them
together is time: the face cluster and the voice cluster that are visible and
speaking at the same moments, consistently, are very likely the same person.

Three stages:

  1. co-occurrence  — pure windowed-overlap arithmetic, no I/O (testable
                       directly against synthetic presence data)
  2. matching       — greedy one-to-one pairing of the strongest overlaps,
                       so one face cluster is never claimed by two identities
  3. fusion         — an identity row per matched pair, an LLM naming
                       attempt from the paired voice's transcript, and
                       IDENTITY_LINK edges from every node that shows the
                       face or carries the voice
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from .config import GraphSettings
from .extraction.claims import SpeakerNameExtractor
from .repository import GraphRepository

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class FaceVoicePair:
    """A candidate identity: one face cluster and one voice cluster that
    co-occurred within a single source file."""

    source_file_id: str
    face_cluster_id: str
    voice_cluster_id: str
    overlap_ratio: float
    shared_windows: int


@dataclass
class IdentityFusionReport:
    identities_created: int = 0
    named: int = 0
    face_links: int = 0
    voice_links: int = 0


# -- stage 1: co-occurrence ---------------------------------------------------


def _windows_covered(intervals: list[tuple[float, float]], window_sec: float) -> set[int]:
    """Every window index a set of [start, end) intervals touches.

    Windows, not raw seconds, because a face cluster's presence comes from
    node-granularity intervals (a segment can be several seconds) while a
    voice cluster's comes from turn-granularity ones (often under a second);
    reducing both to the same window resolution is what makes them
    comparable at all.
    """
    covered: set[int] = set()
    for start, end in intervals:
        if end <= start:
            continue
        first = int(start // window_sec)
        last = int((end - 1e-9) // window_sec)  # end is exclusive
        covered.update(range(first, last + 1))
    return covered


def compute_face_voice_overlap(
    source_file_id: str,
    face_presence: dict[str, list[tuple[float, float]]],
    voice_presence: dict[str, list[tuple[float, float]]],
    settings: GraphSettings,
) -> list[FaceVoicePair]:
    """(time both present) / (time either present) for every face/voice
    cluster pair in one source file, as a Jaccard ratio over window indices —
    exactly time-intersection-over-union once both are expressed in windows
    of equal size. Pure function: no I/O, so the windowing and ratio math is
    directly testable against synthetic presence data.
    """
    face_windows = {
        cluster_id: _windows_covered(intervals, settings.identity_window_sec)
        for cluster_id, intervals in face_presence.items()
    }
    voice_windows = {
        cluster_id: _windows_covered(intervals, settings.identity_window_sec)
        for cluster_id, intervals in voice_presence.items()
    }

    pairs: list[FaceVoicePair] = []
    for face_id, fw in face_windows.items():
        if len(fw) < settings.identity_min_windows:
            continue
        for voice_id, vw in voice_windows.items():
            if len(vw) < settings.identity_min_windows:
                continue
            shared = len(fw & vw)
            union = len(fw | vw)
            if union == 0 or shared == 0:
                continue
            pairs.append(FaceVoicePair(source_file_id, face_id, voice_id, shared / union, shared))
    return pairs


# -- stage 2: matching ---------------------------------------------------------


def select_identity_pairs(
    pairs: list[FaceVoicePair], threshold: float
) -> list[FaceVoicePair]:
    """Greedy one-to-one matching: the strongest-overlap pairs win first, and
    a face or voice cluster already claimed cannot be claimed again.

    Without this, a face cluster that co-occurs above the threshold with two
    different voice clusters — plausible if it appears across two source
    files with different recording setups — would spawn two identities for
    one person instead of the single best-supported one.
    """
    claimed_faces: set[str] = set()
    claimed_voices: set[str] = set()
    selected: list[FaceVoicePair] = []

    for pair in sorted(pairs, key=lambda p: p.overlap_ratio, reverse=True):
        if pair.overlap_ratio <= threshold:
            break  # sorted descending: nothing after this clears the bar either
        if pair.face_cluster_id in claimed_faces or pair.voice_cluster_id in claimed_voices:
            continue
        claimed_faces.add(pair.face_cluster_id)
        claimed_voices.add(pair.voice_cluster_id)
        selected.append(pair)

    return selected


# -- stage 3: fusion ------------------------------------------------------------


def _name_identity(
    repository: GraphRepository, name_extractor: SpeakerNameExtractor | None,
    case_id: str, pair: FaceVoicePair, settings: GraphSettings,
) -> str | None:
    if not settings.enable_identity_naming or name_extractor is None or not name_extractor.available:
        return None

    for snippet in repository.fetch_transcript_for_voice_cluster(case_id, pair.voice_cluster_id):
        name = name_extractor.extract(snippet)
        if name:
            return name
    return None


def build_identities(
    repository: GraphRepository,
    name_extractor: SpeakerNameExtractor | None,
    case_id: str,
    settings: GraphSettings,
) -> IdentityFusionReport:
    """Fuse co-occurring face and voice clusters into identities, and link
    every node that shows the face or carries the voice to them."""
    report = IdentityFusionReport()
    repository.clear_identities(case_id)  # idempotent: re-fuse, don't accumulate

    sources = repository.fetch_sources_with_faces_and_voices(case_id)
    log.info("identity fusion: %d source(s) with both faces and voices", len(sources))

    all_pairs: list[FaceVoicePair] = []
    for source_file_id in sources:
        face_presence = repository.fetch_face_presence_by_source(source_file_id)
        voice_presence = repository.fetch_voice_presence_by_source(source_file_id)
        all_pairs.extend(
            compute_face_voice_overlap(source_file_id, face_presence, voice_presence, settings)
        )

    matched = select_identity_pairs(all_pairs, settings.identity_overlap_threshold)
    log.info(
        "identity fusion: %d candidate pair(s), %d matched above %.0f%% overlap",
        len(all_pairs), len(matched), settings.identity_overlap_threshold * 100,
    )

    for pair in matched:
        display_name = _name_identity(repository, name_extractor, case_id, pair, settings)
        identity_id = repository.create_identity(
            case_id, display_name,
            metadata={
                "face_cluster_id": pair.face_cluster_id,
                "voice_cluster_id": pair.voice_cluster_id,
                "source_file_id": pair.source_file_id,
                "overlap_ratio": round(pair.overlap_ratio, 4),
                "shared_windows": pair.shared_windows,
            },
        )
        repository.link_face_cluster_to_identity(pair.face_cluster_id, identity_id)
        repository.link_voice_cluster_to_identity(pair.voice_cluster_id, identity_id)
        report.identities_created += 1
        if display_name:
            report.named += 1

        for node_id in repository.fetch_face_cluster_nodes(case_id, pair.face_cluster_id):
            if repository.insert_identity_link(
                case_id, node_id, identity_id, "face", pair.overlap_ratio
            ):
                report.face_links += 1

        for node_id in repository.fetch_nodes_overlapping_voice_cluster(
            case_id, pair.voice_cluster_id
        ):
            if repository.insert_identity_link(
                case_id, node_id, identity_id, "voice", pair.overlap_ratio
            ):
                report.voice_links += 1

    repository.commit()
    log.info(
        "identity fusion: %d identity(ies) (%d named), %d face link(s), %d voice link(s)",
        report.identities_created, report.named, report.face_links, report.voice_links,
    )
    return report
