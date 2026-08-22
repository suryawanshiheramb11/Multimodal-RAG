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

    similarity_threshold: float = Field(default=0.8, gt=0, le=1.0)
    #: Caps the O(n^2) similarity pass per case; a few hundred image-bearing
    #: nodes is already a lot of pairwise comparisons on CPU.
    max_nodes_for_similarity: int = Field(default=2000, gt=0)

    #: ALIGNS_WITH: same source file, time windows overlapping by at least
    #: this many seconds (0 = any overlap counts).
    min_overlap_sec: float = Field(default=0.0, ge=0)

    face_model_pack: str = "buffalo_l"
    face_detection_confidence: float = Field(default=0.5, gt=0, le=1.0)
    face_embedding_dim: int = Field(default=512, gt=0)
    #: DBSCAN over cosine distance: eps is a distance, not a similarity, so a
    #: *smaller* value means faces must be *more* alike to share a cluster.
    face_cluster_eps: float = Field(default=0.4, gt=0)
    face_cluster_min_samples: int = Field(default=2, gt=0)

    enable_entity_extraction: bool = True
    enable_temporal_alignment: bool = True
    enable_similarity_edges: bool = True
    enable_face_detection: bool = True
    enable_face_clustering: bool = True
