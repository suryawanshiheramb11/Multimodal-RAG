"""Vision-language captioning through a local ollama server.

Talks to ollama's HTTP API directly rather than through a client library: the
call is a single POST, and going direct keeps the timeout, the error surface,
and the base64 handling explicit and easy to reason about.
"""
from __future__ import annotations

import base64
import io
import logging
from pathlib import Path

import httpx

from .base import LazyModel

log = logging.getLogger(__name__)

# Register HEIF / HEIC opener with Pillow if available
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except Exception:
    pass


def encode_image_for_vlm(image_path: Path, max_dimension: int = 1024) -> str:
    """Normalize and encode an image as standard JPEG base64 for vision-language models.

    Ollama / Qwen2.5-VL only natively decodes standard JPEG/PNG bytes. Directly
    reading raw bytes of HEIC, TIFF, or CMYK images causes Ollama 500 errors.
    This function converts any image to RGB JPEG and downscales if needed.
    """
    from PIL import Image

    try:
        with Image.open(image_path) as img:
            # Convert to standard RGB (handles RGBA, P, CMYK, HEIC, etc.)
            rgb_img = img.convert("RGB")

            # Downscale if excessively large to keep inference fast and within memory
            w, h = rgb_img.size
            if max(w, h) > max_dimension:
                scale = max_dimension / max(w, h)
                new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
                rgb_img = rgb_img.resize(new_size, Image.Resampling.LANCZOS)

            buffer = io.BytesIO()
            rgb_img.save(buffer, format="JPEG", quality=85, optimize=True)
            return base64.b64encode(buffer.getvalue()).decode("ascii")
    except Exception as exc:
        log.debug("PIL normalization failed for %s (%s), falling back to raw bytes", image_path.name, exc)
        return base64.b64encode(image_path.read_bytes()).decode("ascii")


class Captioner(LazyModel):
    """Describes an image with a multimodal model served by ollama."""

    def __init__(
        self,
        model_name: str,
        host: str,
        timeout_sec: int,
        *,
        keep_alive: str = "30m",
        max_tokens: int = 60,
        max_image_side: int = 1024,
    ) -> None:
        super().__init__()
        self.name = f"ollama({model_name})"
        self._model_name = model_name
        self._host = host.rstrip("/")
        self._timeout_sec = timeout_sec
        self._keep_alive = keep_alive
        self._max_tokens = max_tokens
        self._max_image_side = max_image_side

    def _build(self):
        """Verify the server is up and the model is actually pulled.

        Checking at load time turns "every caption silently empty" into one
        clear unavailable-reason in the run report.
        """
        response = httpx.get(f"{self._host}/api/tags", timeout=10)
        response.raise_for_status()
        available = {m.get("name", "") for m in response.json().get("models", [])}

        # ollama reports "qwen2.5vl:7b"; accept a bare name matching any tag.
        if not any(
            tag == self._model_name or tag.split(":")[0] == self._model_name.split(":")[0]
            for tag in available
        ):
            raise RuntimeError(
                f"model '{self._model_name}' not pulled on {self._host} "
                f"(available: {sorted(available) or 'none'}); "
                f"run: ollama pull {self._model_name}"
            )

        client = httpx.Client(timeout=self._timeout_sec)
        # Warm the weights during the availability pass, alongside the other
        # model loads, rather than inside the first node's caption call.
        try:
            client.post(
                f"{self._host}/api/generate",
                json={
                    "model": self._model_name,
                    "keep_alive": self._keep_alive,
                    "prompt": "",
                },
            )
        except httpx.HTTPError as exc:
            log.debug("could not pre-warm %s: %s", self._model_name, exc)
        return client

    def caption(self, image_path: Path, prompt: str) -> str | None:
        client = self.load()
        if client is None or not image_path.is_file():
            return None

        try:
            encoded = encode_image_for_vlm(image_path, self._max_image_side)
            response = client.post(
                f"{self._host}/api/generate",
                json={
                    "model": self._model_name,
                    "prompt": prompt,
                    "images": [encoded],
                    "stream": False,
                    # Without keep_alive ollama evicts the model after 5
                    # minutes, and the next node pays a ~10s reload of 13.8GB.
                    "keep_alive": self._keep_alive,
                    "options": {
                        "num_predict": self._max_tokens,
                        "temperature": 0.2,
                    },
                },
            )
            response.raise_for_status()
            text = (response.json().get("response") or "").strip()
        except (httpx.HTTPError, ValueError, OSError) as exc:
            log.warning("captioning failed for %s: %s", image_path.name, exc)
            return None

        return text or None

    def complete(self, prompt: str, *, json_mode: bool = False, max_tokens: int = 128) -> str | None:
        """Plain text completion, used by later phases for entity extraction."""
        client = self.load()
        if client is None:
            return None

        payload: dict = {
            "model": self._model_name,
            "prompt": prompt,
            "stream": False,
            "keep_alive": self._keep_alive,
            "options": {"num_predict": max_tokens, "temperature": 0.1},
        }
        if json_mode:
            payload["format"] = "json"

        try:
            response = client.post(f"{self._host}/api/generate", json=payload)
            response.raise_for_status()
            return (response.json().get("response") or "").strip() or None
        except (httpx.HTTPError, ValueError) as exc:
            log.warning("completion failed: %s", exc)
            return None
