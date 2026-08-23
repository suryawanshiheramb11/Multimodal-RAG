"""CLIP: image embeddings, text embeddings, and zero-shot violence scoring (GPU-accelerated)."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .base import LazyModel, get_device

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ViolenceScore:
    """Zero-shot violence assessment averaged over the frames of a node."""

    score: float
    frames_scored: int
    #: Mean probability per prompt, so a reviewer can see *why* the score is
    #: what it is rather than trusting one opaque number.
    per_prompt: dict[str, float]


class ClipEncoder(LazyModel):
    """CLIP ViT-B/32 — 512-d joint image/text space (GPU-accelerated)."""

    def __init__(
        self,
        model_name: str,
        expected_dim: int,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.name = f"clip({model_name})"
        self._model_name = model_name
        self._expected_dim = expected_dim
        self._processor = None
        self._device = device or get_device()
        #: Prompt features keyed by the prompt tuple. The violence prompts are
        #: identical for every node in a run, so encoding them per node repeats
        #: the same text forward pass hundreds of times for the same answer.
        self._prompt_cache: dict[tuple[str, ...], torch.Tensor] = {}

    def _build(self):
        from transformers import CLIPModel, CLIPProcessor

        model = CLIPModel.from_pretrained(self._model_name)
        model.eval()
        model.to(self._device)  # Move to GPU/MPS
        if model.config.projection_dim != self._expected_dim:
            raise ValueError(
                f"{self._model_name} projects to {model.config.projection_dim}-d "
                f"but the schema column expects {self._expected_dim}"
            )
        self._processor = CLIPProcessor.from_pretrained(self._model_name)
        log.info("CLIP on device %s", self._device)
        return model

    # -- embeddings ---------------------------------------------------------

    def embed_image(self, image_path: Path) -> np.ndarray | None:
        model = self.load()
        if model is None:
            return None
        image = self._open(image_path)
        if image is None:
            return None

        inputs = self._processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}  # Move to GPU/MPS
        with torch.no_grad():
            features = model.get_image_features(**inputs)
        return self._normalise(features)[0]

    def embed_text(self, text: str) -> np.ndarray | None:
        model = self.load()
        if model is None or not text or not text.strip():
            return None

        # CLIP's text encoder truncates at 77 tokens; truncation is explicit
        # here so a long transcript degrades predictably instead of erroring.
        inputs = self._processor(
            text=[text], return_tensors="pt", padding=True, truncation=True, max_length=77
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}  # Move to GPU/MPS
        with torch.no_grad():
            features = model.get_text_features(**inputs)
        return self._normalise(features)[0]

    # -- zero-shot ----------------------------------------------------------

    def _prompt_features(self, model, prompts: list[str]) -> torch.Tensor:
        """L2-normalised text features for `prompts`, encoded once per run."""
        key = tuple(prompts)
        cached = self._prompt_cache.get(key)
        if cached is not None:
            return cached

        inputs = self._processor(
            text=list(prompts), return_tensors="pt", padding=True,
            truncation=True, max_length=77,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            features = self._unwrap(model.get_text_features(**inputs))
        features = features / features.norm(p=2, dim=-1, keepdim=True)
        self._prompt_cache[key] = features
        return features

    def score_violence(
        self, image_paths: list[Path], prompts: list[str], violent_count: int
    ) -> ViolenceScore | None:
        """Softmax across all prompts per frame, then average across frames.

        The score is the probability mass on the violent prompts, so it stays
        in [0, 1] and is comparable between nodes.

        The text side is cached: the prompts never vary within a run, so only
        the images need a forward pass. That also means the logits are computed
        here rather than by `model(**inputs)` — same arithmetic CLIP does
        internally (cosine similarity scaled by the learned temperature).
        """
        model = self.load()
        if model is None or not image_paths or not prompts:
            return None

        images = [img for img in (self._open(p) for p in image_paths) if img is not None]
        if not images:
            return None

        text_features = self._prompt_features(model, prompts)

        inputs = self._processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}  # Move to GPU/MPS
        with torch.no_grad():
            image_features = self._unwrap(model.get_image_features(**inputs))
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            # (n_images, n_prompts), matching CLIPModel.logits_per_image.
            logits = model.logit_scale.exp() * image_features @ text_features.t()
            probabilities = logits.softmax(dim=1).cpu().numpy()

        mean_probabilities = probabilities.mean(axis=0)
        return ViolenceScore(
            score=float(mean_probabilities[:violent_count].sum()),
            frames_scored=len(images),
            per_prompt={
                prompt: round(float(value), 4)
                for prompt, value in zip(prompts, mean_probabilities, strict=True)
            },
        )

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _open(path: Path):
        from PIL import Image, UnidentifiedImageError

        try:
            with Image.open(path) as image:
                return image.convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            log.warning("cannot open %s for CLIP: %s", path, exc)
            return None

    @staticmethod
    def _unwrap(features):
        """Extract the projected embedding tensor.

        transformers >=5 has `get_image_features`/`get_text_features` return a
        `BaseModelOutputWithPooling` (or a bare tuple) instead of the plain
        tensor older examples assume; `pooler_output` is that projected
        embedding — the same 512-d vector the tensor-returning API used to
        hand back directly.
        """
        if hasattr(features, "pooler_output"):
            return features.pooler_output
        if isinstance(features, tuple):
            return features[0]
        return features

    def _normalise(self, features) -> np.ndarray:
        features = self._unwrap(features)
        features = features / features.norm(p=2, dim=-1, keepdim=True)
        return features.cpu().numpy().astype(np.float32)
