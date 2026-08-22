"""Enrichment configuration: model identifiers, prompts, and work limits.

Model names are configurable so a demo can swap `large-v3-turbo` for `base`
without touching code, and so a reviewer can see exactly which checkpoint
produced a given embedding.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

#: Zero-shot prompts for CLIP violence scoring. The first
#: `violence_prompt_count` entries are the violent classes; the score is the
#: summed probability mass over them after a softmax across all prompts.
DEFAULT_VIOLENCE_PROMPTS = [
    "a violent fight",
    "a person holding a weapon",
    "blood or injury",
    "a peaceful everyday scene",
    "people talking normally",
]

DEFAULT_CAPTION_PROMPT = (
    "Describe this image in detail, including any violence, weapons, or injuries."
)


class ModelNames(BaseModel):
    model_config = {"extra": "forbid", "protected_namespaces": ()}

    asr: str = "large-v3-turbo"
    audio_events: str = "MIT/ast-finetuned-audioset-10-10-0.4593"
    clip: str = "openai/clip-vit-base-patch32"
    detector: str = "yolov8s.pt"
    captioner: str = "qwen2.5vl:7b"
    text_encoder: str = "all-MiniLM-L6-v2"


class EnrichmentSettings(BaseModel):
    """Everything the feature-extraction phase needs to know."""

    model_config = {"extra": "forbid", "protected_namespaces": ()}

    device: str = "cpu"
    compute_type: str = "int8"
    models: ModelNames = ModelNames()

    ollama_host: str = "http://localhost:11434"
    ollama_timeout_sec: int = Field(default=180, gt=0)

    violence_prompts: list[str] = Field(default_factory=lambda: list(DEFAULT_VIOLENCE_PROMPTS))
    #: How many leading prompts count as "violent" when summing probabilities.
    violence_prompt_count: int = Field(default=3, gt=0)
    caption_prompt: str = DEFAULT_CAPTION_PROMPT

    detection_confidence: float = Field(default=0.25, gt=0, le=1.0)
    #: Per-node ceiling on frames handed to CLIP and YOLO. Enrichment is the
    #: expensive phase; without this a long segment multiplies model calls.
    max_frames_analyzed: int = Field(default=8, gt=0)
    ocr_language: str = "en"

    #: Pages whose embedded text is shorter than this are treated as scanned
    #: images and sent through OCR.
    ocr_page_text_threshold: int = Field(default=40, ge=0)

    #: Expected embedding widths. Checked against each model's real output at
    #: runtime: a silent dimension mismatch would corrupt every similarity
    #: search downstream, so it fails loudly instead.
    text_embedding_dim: int = Field(default=384, gt=0)
    clip_embedding_dim: int = Field(default=512, gt=0)
    audio_embedding_dim: int = Field(default=768, gt=0)

    #: Feature switches, so a run can skip a slow or unavailable stage.
    enable_asr: bool = True
    enable_audio_events: bool = True
    enable_violence: bool = True
    enable_detection: bool = True
    enable_caption: bool = True
    enable_ocr: bool = True

    @property
    def violent_prompts(self) -> list[str]:
        return self.violence_prompts[: self.violence_prompt_count]
