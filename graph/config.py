"""Graph-construction configuration."""
from __future__ import annotations

from pydantic import BaseModel, Field

#: Canonical entity types the extraction prompt is constrained to, per the
#: phase 3 spec. Anything the model returns outside this set is kept as
#: 'other' rather than silently dropped.
ENTITY_TYPES = ("person", "weapon", "location", "phone", "vehicle", "organization")

#: COCO class -> entity type, for turning YOLO detections (already computed in
#: phase 2) into entities without a second model call.
DETECTION_ENTITY_TYPES: dict[str, str] = {
    "person": "person",
    "knife": "weapon",
    "scissors": "weapon",
    "baseball bat": "weapon",
    "gun": "weapon",
    "pistol": "weapon",
    "rifle": "weapon",
    "cell phone": "phone",
    "car": "vehicle",
    "truck": "vehicle",
    "bus": "vehicle",
    "motorcycle": "vehicle",
    "bicycle": "vehicle",
}

ENTITY_EXTRACTION_PROMPT = """\
Extract entities (persons, weapons, locations, phones, vehicles, organizations) \
from this text. Return JSON.

Respond with ONLY a JSON object of this exact shape, no other text:
{{"entities": [{{"type": "person|weapon|location|phone|vehicle|organization", \
"name": "..."}}]}}
If there are no entities, return {{"entities": []}}.

Text:
\"\"\"
{text}
\"\"\"
"""

#: Fed a handful of text snippets from evidence that occurred within the same
#: short time window; asked for one factual sentence, not a story, so it
#: stays a label rather than something a reviewer has to fact-check on its own.
TIMELINE_EVENT_PROMPT = """\
The following pieces of evidence occurred within a short time window of each \
other and may describe the same event. Summarize what is happening in one or \
two short factual sentences. Do not speculate beyond what is stated. Respond \
with plain text only, no preamble.

Evidence:
{snippets}
"""

#: Reduces a node's fused text (transcript + caption + OCR, or page text) to
#: one comparable assertion. Contradiction detection needs claims, not prose:
#: "Transcript: ... Visual description: ..." blocks from two nodes cannot be
#: compared reliably, whereas two subject-verb-object sentences can.
#: The explicit NONE escape hatch stops the model inventing a claim for text
#: that asserts nothing (a caption of an empty room, a page of headers).
CLAIM_EXTRACTION_PROMPT = """\
Extract the main factual claim from this text as a simple subject-verb-object \
statement.

Respond with ONLY the claim, as one short sentence. No preamble, no quotes, \
no explanation.
If the text states no factual claim, respond with exactly: NONE

Text:
\"\"\"
{text}
\"\"\"
"""

#: Judges two extracted claims. `unrelated` is offered as an explicit option
#: (rather than leaving the model to choose between only contradicts and
#: corroborates) because most linked pairs genuinely are unrelated, and a
#: forced binary choice would manufacture disagreements that aren't there.
CONTRADICTION_PROMPT = """\
Claim A: {claim_a}
Claim B: {claim_b}

Do these statements contradict, corroborate, or are they unrelated?

Respond with ONLY a JSON object of this exact shape, no other text:
{{"relation": "contradicts|corroborates|unrelated", "confidence": 0.0, \
"explanation": "..."}}
where confidence is a number between 0.0 and 1.0, and explanation is one \
short sentence.
"""

#: The verdicts that become stored edges. `unrelated` is a real answer, not a
#: failure — it just isn't worth an edge.
RELATION_EDGE_TYPES: dict[str, str] = {
    "contradicts": "CONTRADICTS",
    "corroborates": "CORROBORATES",
}

#: Fed the transcript of a voice cluster that has just been fused with a face
#: cluster (i.e. a specific person's speech), asked for the name if the
#: person identified themselves or was addressed by name. The NONE escape
#: hatch matters more here than anywhere else in this file: most transcripts
#: never name their speaker, and a model that invents a name from tone or
#: guesswork would poison an identity's display name with something false.
NAME_EXTRACTION_PROMPT = """\
The following is a transcript of one person speaking. If it states or \
implies that person's own name — for example "Hi, this is John", "John \
speaking", someone addressing them by name, or them signing off with a name \
— respond with ONLY that name, nothing else.

If no name for this speaker is stated or implied, respond with exactly: NONE

Transcript:
\"\"\"
{text}
\"\"\"
"""

#: The only LLM call in question answering, and deliberately the last step:
#: classification and retrieval (graph/qa.py) never touch a model — they are
#: plain keyword matching and SQL against the graph already built by phases
#: 3-7. By the time this prompt runs, every fact it can cite already exists;
#: its only job is turning that fixed list into one readable sentence, which
#: is why "say so plainly" is spelled out rather than left implicit — a model
#: padding a thin fact list with plausible-sounding filler is the one failure
#: mode this whole design exists to avoid.
ANSWER_SYNTHESIS_PROMPT = """\
Answer the question using ONLY the evidence listed below. Do not state \
anything the evidence does not support. If the evidence does not answer the \
question, say so plainly instead of guessing.

Question: {question}

Evidence:
{facts}

Respond with a short, direct answer (2-4 sentences). Reference specific \
evidence — a timestamp, a page, a name — where relevant.
"""


class GraphSettings(BaseModel):
    model_config = {"extra": "forbid", "protected_namespaces": ()}

    ollama_host: str = "http://localhost:11434"
    ollama_timeout_sec: int = Field(default=180, gt=0)
    #: Reuses the vision-language model already pulled for phase 2 captioning
    #: rather than pulling a second multi-gigabyte checkpoint; it handles
    #: text-only prompts fine.
    entity_model: str = "qwen2.5vl:7b"
    #: Text handed to the LLM is capped: a multi-page transcript would blow
    #: past the model's context and inflate latency for no extra entities.
    max_extraction_chars: int = Field(default=4000, gt=0)

    text_encoder_model: str = "all-MiniLM-L6-v2"
    entity_embedding_dim: int = Field(default=384, gt=0)

    #: Same CLIP checkpoint enrichment used to fill clip_embedding — reusing
    #: it (rather than embedding with something else) is what makes a graph
    #: step's on-the-fly text vector comparable to the stored image vectors.
    clip_model: str = "openai/clip-vit-base-patch32"
    clip_embedding_dim: int = Field(default=512, gt=0)

    similarity_threshold: float = Field(default=0.8, gt=0, le=1.0)
    #: Caps the O(n^2) similarity pass per case; a few hundred image-bearing
    #: nodes is already a lot of pairwise comparisons on CPU.
    max_nodes_for_similarity: int = Field(default=2000, gt=0)

    #: ALIGNS_WITH: same source file, time windows overlapping by at least
    #: this many seconds (0 = any overlap counts).
    min_overlap_sec: float = Field(default=0.0, ge=0)

    #: REFERENCES: pdf page image <-> video segment's representative frame,
    #: both already-stored CLIP image vectors compared entirely in SQL.
    reference_similarity_threshold: float = Field(default=0.7, gt=0, le=1.0)
    #: DESCRIBES: a transcript's CLIP *text* vector (computed on the fly,
    #: never persisted) against a frame's stored CLIP *image* vector. Text/
    #: image cosine similarity runs much lower than image/image, hence the
    #: separate, lower threshold from `reference_similarity_threshold`.
    describes_similarity_threshold: float = Field(default=0.3, gt=0, le=1.0)
    #: Caps the page x segment join (REFERENCES) and the transcript x frame
    #: pass (DESCRIBES), same rationale as max_nodes_for_similarity.
    max_nodes_for_cross_modal: int = Field(default=2000, gt=0)

    #: Timeline event grouping: nodes within this many seconds of each other
    #: (by start_time) are candidates for the same event.
    timeline_window_sec: float = Field(default=30.0, gt=0)
    #: MiniLM text_embedding cosine similarity counted as "high" for grouping
    #: two nodes into the same event when they share no entity.
    timeline_text_similarity_threshold: float = Field(default=0.75, gt=0, le=1.0)
    #: Reuses the same local model as entity extraction — per its own comment,
    #: it handles text-only prompts fine, so no second model needs pulling.
    timeline_event_model: str = "qwen2.5vl:7b"
    #: A group whose LLM summary is skipped or fails still gets a event row,
    #: with a plain templated description instead of an aborted run.
    enable_timeline_llm_summary: bool = True

    face_model_pack: str = "buffalo_l"
    face_detection_confidence: float = Field(default=0.5, gt=0, le=1.0)
    face_embedding_dim: int = Field(default=512, gt=0)
    #: DBSCAN over cosine distance: eps is a distance, not a similarity, so a
    #: *smaller* value means faces must be *more* alike to share a cluster.
    face_cluster_eps: float = Field(default=0.4, gt=0)
    face_cluster_min_samples: int = Field(default=2, gt=0)

    #: Claim extraction: which node types carry an assertion worth distilling.
    #: An 'image' node's text is a caption of what is visible, which is exactly
    #: the kind of claim a witness statement can contradict, so it is included.
    claim_node_types: list[str] = Field(
        default_factory=lambda: ["scene_segment", "audio_track", "page", "image"]
    )
    #: Same ceiling and rationale as max_extraction_chars: a full page of text
    #: would blow past the model's context for no better a one-line claim.
    max_claim_chars: int = Field(default=4000, gt=0)

    #: Which existing edge types contribute candidate pairs. ALIGNS_WITH is the
    #: temporal link, DESCRIBES the spoken-to-visual one, and REFERENCES the
    #: document-to-video one — a claim can conflict with evidence reached by
    #: any of the three.
    contradiction_alignment_types: list[str] = Field(
        default_factory=lambda: ["ALIGNS_WITH", "DESCRIBES", "REFERENCES"]
    )
    #: Pre-filter before spending an LLM call: two nodes whose MiniLM text
    #: vectors are this dissimilar are talking about different things, and
    #: cannot contradict each other. Deliberately low — a denial ("no weapon")
    #: and an observation ("a knife is visible") are opposites in meaning but
    #: still share enough vocabulary to clear a low bar.
    contradiction_similarity_threshold: float = Field(default=0.3, ge=0, le=1.0)
    #: Used when the model omits a usable confidence of its own.
    contradiction_default_confidence: float = Field(default=0.5, gt=0, le=1.0)
    #: A very common entity ("person") can be mentioned by every node in a
    #: case; without this its pair count alone is O(n^2).
    max_nodes_per_entity_for_pairs: int = Field(default=25, gt=1)
    #: Ceiling on LLM comparisons per run, applied after the similarity
    #: pre-filter — that count is what actually costs minutes of inference.
    max_contradiction_pairs: int = Field(default=500, gt=0)

    #: pyannote's community diarization pipeline. Like every pretrained
    #: pyannote checkpoint it is gated on Hugging Face: a token with the
    #: model's terms accepted is required (free, but a manual step) — see
    #: SpeakerDiarizer's docstring. Never set the token here; it is read from
    #: HF_TOKEN/PYANNOTE_AUTH_TOKEN at the composition root and passed in
    #: directly, the same env-only rule DatabaseSettings applies to the DB
    #: password, so a credential can never end up in a settings dump or log.
    diarization_model: str = "pyannote/speaker-diarization-community-1"
    #: WeSpeaker ResNet34, confirmed by loading it: see the phase 7 schema
    #: migration for how this number was obtained.
    voice_embedding_dim: int = Field(default=256, gt=0)
    #: Agglomerative clustering over cosine distance: like face_cluster_eps,
    #: a *smaller* value means turns must be *more* alike to share a cluster.
    voice_cluster_distance_threshold: float = Field(default=0.4, gt=0)
    #: A single stray turn should not become its own "cluster" that a face
    #: never gets the chance to match against.
    voice_cluster_min_segments: int = Field(default=2, gt=0)

    #: Co-occurrence is measured in windows this wide: 1-second granularity
    #: matches the frame sample rate frames are extracted at, so a face's
    #: presence and a voice's presence are compared at the same resolution
    #: either was actually observed.
    identity_window_sec: float = Field(default=1.0, gt=0)
    #: (time both present) / (time either present) must clear this to link a
    #: face cluster and a voice cluster into one identity.
    identity_overlap_threshold: float = Field(default=0.6, gt=0, le=1.0)
    #: "Consistent" co-occurrence, per the spec: a single overlapping second
    #: could be coincidence (someone glances into frame while another person
    #: speaks off-camera); this many is a pattern.
    identity_min_windows: int = Field(default=3, gt=0)
    #: Fused identities still get an LLM naming attempt disabled independently
    #: of contradiction/timeline LLM use, so a case with ollama down loses
    #: names but not the identity fusion itself.
    enable_identity_naming: bool = True

    #: Cap on how many retrieved facts get handed to the LLM (or, when the
    #: model is unavailable, listed in the templated fallback answer) — a
    #: broad question like "tell me about the meeting" can legitimately
    #: retrieve dozens of nodes, and the prompt shouldn't grow without bound.
    max_qa_facts: int = Field(default=10, gt=0)
    #: When false, or when the model is unavailable, an answer is still
    #: produced — a plain listing of the retrieved facts with no LLM call at
    #: all. This is the "don't use much AI" half of the design: retrieval
    #: always works without a model; only the prose wording depends on one.
    enable_qa_llm_synthesis: bool = True

    enable_entity_extraction: bool = True
    enable_temporal_alignment: bool = True
    enable_similarity_edges: bool = True
    enable_face_detection: bool = True
    enable_face_clustering: bool = True
    enable_document_video_linking: bool = True
    enable_transcript_visual_linking: bool = True
    enable_timeline_events: bool = True
    enable_claim_extraction: bool = True
    enable_contradiction_detection: bool = True
    enable_voice_diarization: bool = True
    enable_voice_clustering: bool = True
    enable_identity_fusion: bool = True
