"""Text extraction from images with PaddleOCR under strict process isolation.

PaddlePaddle and PyTorch on macOS / Apple Silicon use conflicting OpenMP / C++
threading and memory allocators. To prevent crashes, deadlocks, and memory
corruption, PaddleOCR is isolated inside a dedicated worker process spawned via
`multiprocessing.get_context('spawn')`.

Heavy image arrays use `multiprocessing.shared_memory` for zero-copy IPC,
while lightweight commands and results travel over IPC Queues.
"""
from __future__ import annotations

import atexit
import itertools
import logging
import multiprocessing as mp
import queue as queue_mod
import time
from dataclasses import dataclass
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np

from .base import LazyModel

log = logging.getLogger(__name__)

#: How often an idle worker wakes to check that its parent is still alive.
_ORPHAN_CHECK_SEC = 5.0

#: Ceiling on one OCR call. A frame that exceeds it is abandoned, not waited on.
_READ_TIMEOUT_SEC = 60.0


@dataclass(frozen=True)
class OcrResult:
    text: str
    line_count: int
    mean_confidence: float | None


def _paddle_ocr_worker(language: str, req_queue: mp.Queue, res_queue: mp.Queue) -> None:
    """Isolated worker process for PaddleOCR."""
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["FLAGS_fraction_of_gpu_memory_to_use"] = "0.0"

    parent_pid = os.getppid()

    try:
        import paddle
        if hasattr(paddle, "set_num_threads"):
            paddle.set_num_threads(1)
    except Exception:
        pass

    try:
        from paddleocr import PaddleOCR
        try:
            model = PaddleOCR(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                lang=language,
            )
            uses_legacy_api = False
        except (TypeError, ValueError):
            model = PaddleOCR(use_angle_cls=False, lang=language)
            uses_legacy_api = True
        res_queue.put(("READY", None))
    except Exception as exc:
        res_queue.put(("INIT_ERROR", str(exc)))
        return

    while True:
        try:
            msg = req_queue.get(timeout=_ORPHAN_CHECK_SEC)
        except queue_mod.Empty:
            # The parent is not always shut down politely: uvicorn --reload
            # SIGKILLs the app process on every code change, and a Ctrl-C or an
            # OOM kill does the same. `daemon=True` only covers a *clean*
            # parent exit, so without this check the worker would sit on
            # req_queue forever, orphaned to init and holding PaddleOCR's ~1GB
            # resident. Re-parenting is the signal that nobody is coming back.
            if os.getppid() != parent_pid:
                break
            continue
        except Exception:
            break

        if msg == "STOP":
            break

        req_id, cmd, payload = msg
        try:
            if cmd == "READ_PATH":
                image_path, max_side = payload
                lines = _run_ocr(model, _downscale(image_path, max_side), uses_legacy_api)
                res_queue.put((req_id, "OK", lines))

            elif cmd == "READ_SHM":
                shm_name, shape, dtype_str = payload
                existing_shm = shared_memory.SharedMemory(name=shm_name)
                try:
                    img_array = np.ndarray(shape, dtype=np.dtype(dtype_str), buffer=existing_shm.buf)
                    lines = _run_ocr(model, img_array, uses_legacy_api)
                    res_queue.put((req_id, "OK", lines))
                finally:
                    existing_shm.close()

            elif cmd == "PING":
                res_queue.put((req_id, "PONG", None))
            else:
                res_queue.put((req_id, "ERROR", f"Unknown command: {cmd}"))

        except Exception as exc:
            res_queue.put((req_id, "ERROR", str(exc)))


def _downscale(image_path: str, max_side: int):
    """Open an image, shrinking it to `max_side` if it is larger.

    PaddleOCR's runtime tracks the number of text regions it detects, and an
    oversized frame simply detects more of them: a 3600px screen recording
    costs 26s where the same frame at 2400px costs 12s and still yields 36 of
    the 40 lines. Returns a path unchanged when no resize is needed, so the
    normal case stays a plain file read inside Paddle.
    """
    from PIL import Image

    try:
        with Image.open(image_path) as img:
            width, height = img.size
            if max(width, height) <= max_side:
                return image_path
            scale = max_side / max(width, height)
            resized = img.convert("RGB").resize(
                (max(1, int(width * scale)), max(1, int(height * scale))),
                Image.Resampling.LANCZOS,
            )
            return np.array(resized)
    except Exception as exc:
        log.debug("cannot downscale %s (%s); passing through", image_path, exc)
        return image_path


def _run_ocr(model, image_input, uses_legacy_api: bool) -> list[tuple[str, float | None]]:
    if not uses_legacy_api and hasattr(model, "predict"):
        return _parse_v3(model.predict(image_input))
    return _parse_v2(model.ocr(image_input))


def _parse_v3(results) -> list[tuple[str, float | None]]:
    """3.x returns dict-like records with parallel text/score lists."""
    lines: list[tuple[str, float | None]] = []
    for record in results or []:
        data = record if isinstance(record, dict) else getattr(record, "json", {})
        data = data.get("res", data) if isinstance(data, dict) else {}
        texts = data.get("rec_texts") or []
        scores = data.get("rec_scores") or [None] * len(texts)
        lines.extend(zip(texts, scores, strict=False))
    return lines


def _parse_v2(results) -> list[tuple[str, float | None]]:
    """2.x returns [[ [box, (text, score)], ... ]] per image."""
    lines: list[tuple[str, float | None]] = []
    for page in results or []:
        for entry in page or []:
            if not entry or len(entry) < 2:
                continue
            payload = entry[1]
            if isinstance(payload, (list, tuple)) and payload:
                lines.append((str(payload[0]), float(payload[1]) if len(payload) > 1 else None))
    return lines


class OcrReader(LazyModel):
    """PaddleOCR text detection + recognition with strict process isolation."""

    def __init__(self, language: str = "en", max_side: int = 2400) -> None:
        super().__init__()
        self.name = f"paddleocr({language})"
        self._language = language
        self._max_side = max_side
        self._ctx = mp.get_context("spawn")
        self._proc: mp.Process | None = None
        self._req_q: mp.Queue | None = None
        self._res_q: mp.Queue | None = None
        self._ids = itertools.count(1)
        atexit.register(self.close)

    def _build(self):
        """Spawns an isolated PaddleOCR process and waits for readiness."""
        self._req_q = self._ctx.Queue()
        self._res_q = self._ctx.Queue()

        self._proc = self._ctx.Process(
            target=_paddle_ocr_worker,
            args=(self._language, self._req_q, self._res_q),
            daemon=True,
        )
        self._proc.start()

        # Wait up to 30s for the worker to initialize PaddleOCR
        try:
            status, err = self._res_q.get(timeout=30)
            if status != "READY":
                self.close()
                raise RuntimeError(f"PaddleOCR worker failed to initialize: {err}")
        except Exception as exc:
            self.close()
            raise RuntimeError(f"PaddleOCR worker startup timed out or failed: {exc}") from exc

        return self._proc

    def _request(self, cmd: str, payload) -> list[tuple[str, float | None]] | None:
        """Send one command and wait for *its own* reply.

        Replies carry the id of the request they answer, and anything older is
        discarded on arrival. Without that, a single call that hit the timeout
        would leave its late reply in the queue and every subsequent frame
        would silently receive the previous frame's text.
        """
        req_id = next(self._ids)
        self._req_q.put((req_id, cmd, payload))

        deadline = time.monotonic() + _READ_TIMEOUT_SEC
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"OCR worker did not answer within {_READ_TIMEOUT_SEC:.0f}s")

            got_id, status, result = self._res_q.get(timeout=remaining)
            if got_id != req_id:
                log.debug("discarding stale OCR reply %s", got_id)
                continue
            if status != "OK":
                raise RuntimeError(str(result))
            return result

    def read(self, image_path: Path) -> OcrResult | None:
        """Run OCR on an image file in the isolated worker process."""
        if not image_path.is_file():
            return None

        model = self.load()
        if model is None or self._req_q is None or self._res_q is None:
            return None

        try:
            lines = self._request("READ_PATH", (str(image_path), self._max_side))
        except Exception as exc:
            log.warning("OCR failed for %s: %s", image_path.name, exc)
            return None

        if not lines:
            return OcrResult(text="", line_count=0, mean_confidence=None)

        texts = [text for text, _ in lines]
        confidences = [score for _, score in lines if score is not None]
        return OcrResult(
            text="\n".join(texts),
            line_count=len(texts),
            mean_confidence=(
                round(sum(confidences) / len(confidences), 4) if confidences else None
            ),
        )

    def read_array(self, image_array: np.ndarray) -> OcrResult | None:
        """Zero-copy OCR on an image array via shared memory."""
        model = self.load()
        if model is None or self._req_q is None or self._res_q is None:
            return None

        shm = shared_memory.SharedMemory(create=True, size=image_array.nbytes)
        try:
            shm_array = np.ndarray(image_array.shape, dtype=image_array.dtype, buffer=shm.buf)
            np.copyto(shm_array, image_array)

            lines = self._request(
                "READ_SHM", (shm.name, image_array.shape, str(image_array.dtype))
            )
        except Exception as exc:
            log.warning("OCR shared memory worker error: %s", exc)
            return None
        finally:
            shm.close()
            shm.unlink()

        if not lines:
            return OcrResult(text="", line_count=0, mean_confidence=None)

        texts = [text for text, _ in lines]
        confidences = [score for _, score in lines if score is not None]
        return OcrResult(
            text="\n".join(texts),
            line_count=len(texts),
            mean_confidence=(
                round(sum(confidences) / len(confidences), 4) if confidences else None
            ),
        )

    def close(self) -> None:
        """Cleanly terminate the worker process."""
        if self._req_q is not None:
            try:
                self._req_q.put("STOP")
            except Exception:
                pass
        if self._proc is not None and self._proc.is_alive():
            self._proc.join(timeout=2)
            if self._proc.is_alive():
                # Busy inside PaddleOCR and deaf to STOP: escalate rather than
                # leave a 1GB process behind.
                self._proc.terminate()
                self._proc.join(timeout=5)
            if self._proc.is_alive():
                self._proc.kill()
                self._proc.join(timeout=5)
        self._proc = None
        self._req_q = None
        self._res_q = None

    def __del__(self) -> None:
        self.close()
