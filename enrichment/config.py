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

#: Kept deliberately terse. Generation dominates caption latency (~24ms/token),
#: and an open-ended "describe in detail" spends its whole budget before it
#: reaches the forensic detail the search index actually needs.
DEFAULT_CAPTION_PROMPT = (
    "Describe this image in one sentence. "
    "Name any weapon, violence, injury, or visible text."
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
    #: Threads ctranslate2 gives Whisper. 0 means every core: faster-whisper's
    #: own default is 4, which leaves most of an Apple Silicon machine idle
    #: (18.1s vs 13.8s on a 65s track).
    asr_cpu_threads: int = Field(default=0, ge=0)
    #: Windows decoded together by faster-whisper's batched pipeline. Set to 1
    #: to fall back to sequential decoding — batching is ~1.3x faster again but
    #: chooses slightly coarser segment boundaries.
    asr_batch_size: int = Field(default=8, gt=0)
    models: ModelNames = ModelNames()

    ollama_host: str = "http://localhost:11434"
    ollama_timeout_sec: int = Field(default=180, gt=0)
    #: How long ollama keeps the vision model resident between calls. The
    #: default (5m) expires mid-run on a slow case and the next caption pays a
    #: ~10s reload of a 13.8GB model — which is most of what looked like
    #: "captioning is slow".
    ollama_keep_alive: str = "30m"
    #: Ceiling on caption length. Straight latency: see DEFAULT_CAPTION_PROMPT.
    caption_max_tokens: int = Field(default=60, gt=0)
    #: Frames are downscaled to this before being sent to the VLM. Beyond it
    #: the extra vision tokens cost prefill time without changing the caption.
    caption_max_side: int = Field(default=1024, gt=0)

    violence_prompts: list[str] = Field(default_factory=lambda: list(DEFAULT_VIOLENCE_PROMPTS))
    #: How many leading prompts count as "violent" when summing probabilities.
    violence_prompt_count: int = Field(default=3, gt=0)
    caption_prompt: str = DEFAULT_CAPTION_PROMPT

    detection_confidence: float = Field(default=0.25, gt=0, le=1.0)
    #: Per-node ceiling on frames handed to CLIP and YOLO. Enrichment is the
    #: expensive phase; without this a long segment multiplies model calls.
    max_frames_analyzed: int = Field(default=8, gt=0)
    ocr_language: str = "en"
    #: Longest side an image is downscaled to before OCR. PaddleOCR's cost
    #: scales with the number of text lines it detects, and a 3600px screenshot
    #: costs 26s against 12s at 2400px for 36 of the same 40 lines. Raise it to
    #: recover fine print; lower it for speed.
    ocr_max_side: int = Field(default=2400, gt=0)
    #: PaddleOCR model pair. The library defaults to the `medium` checkpoints;
    #: the mobile pair reads this project's frames in half the time and finds
    #: *more* lines (92 vs 83 on a dense screenshot), so it is the default.
    #: Set either to None to fall back to PaddleOCR's own choice for `lang`.
    ocr_det_model: str | None = "PP-OCRv5_mobile_det"
    ocr_rec_model: str | None = "PP-OCRv5_mobile_rec"

    #: Pages whose embedded text is shorter than this are treated as scanned
    #: images and sent through OCR.
    ocr_page_text_threshold: int = Field(default=40, ge=0)

    #: Expected embedding widths. Checked against each model's real output at
    #: runtime: a silent dimension mismatch would corrupt every similarity
    #: search downstream, so it fails loudly instead.
    text_embedding_dim: int = Field(default=384, gt=0)
    clip_embedding_dim: int = Field(default=512, gt=0)
    audio_embedding_dim: int = Field(default=768, gt=0)

    #: Run the out-of-process stages (OCR, and the VLM caption over HTTP)
    #: concurrently with the in-process torch models. Both spend their time
    #: waiting on another process, so overlapping them is close to free.
    parallel_stages: bool = True

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
