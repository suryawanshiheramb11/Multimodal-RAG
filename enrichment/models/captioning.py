"""Vision-language captioning through a local ollama server.

Talks to ollama's HTTP API directly rather than through a client library: the
call is a single POST, and going direct keeps the timeout, the error surface,
and the base64 handling explicit and easy to reason about.
"""
from __future__ import annotations

import base64
import logging
from pathlib import Path

import httpx

from .base import LazyModel

log = logging.getLogger(__name__)


class Captioner(LazyModel):
    """Describes an image with a multimodal model served by ollama."""

    def __init__(self, model_name: str, host: str, timeout_sec: int) -> None:
        super().__init__()
        self.name = f"ollama({model_name})"
        self._model_name = model_name
        self._host = host.rstrip("/")
        self._timeout_sec = timeout_sec

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
        return httpx.Client(timeout=self._timeout_sec)

    def caption(self, image_path: Path, prompt: str) -> str | None:
        client = self.load()
        if client is None or not image_path.is_file():
            return None

        try:
            encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
            response = client.post(
                f"{self._host}/api/generate",
                json={
                    "model": self._model_name,
                    "prompt": prompt,
                    "images": [encoded],
                    "stream": False,
                },
            )
            response.raise_for_status()
            text = (response.json().get("response") or "").strip()
        except (httpx.HTTPError, ValueError, OSError) as exc:
            log.warning("captioning failed for %s: %s", image_path.name, exc)
            return None

        return text or None

    def complete(self, prompt: str, *, json_mode: bool = False) -> str | None:
        """Plain text completion, used by later phases for entity extraction."""
        client = self.load()
        if client is None:
            return None

        payload: dict = {"model": self._model_name, "prompt": prompt, "stream": False}
        if json_mode:
            payload["format"] = "json"

        try:
            response = client.post(f"{self._host}/api/generate", json=payload)
            response.raise_for_status()
            return (response.json().get("response") or "").strip() or None
        except (httpx.HTTPError, ValueError) as exc:
            log.warning("completion failed: %s", exc)
            return None
