"""Thin wrappers around each ML model, all lazily loaded and all optional."""
from .asr import Transcriber, Transcript, TranscriptSegment
from .audio_events import AudioAnalysis, AudioEvent, AudioEventClassifier
from .base import LazyModel
from .captioning import Captioner
from .clip import ClipEncoder, ViolenceScore
from .detection import Detection, ObjectDetector
from .ocr import OcrReader, OcrResult
from .text import TextEncoder

__all__ = [
    "AudioAnalysis",
    "AudioEvent",
    "AudioEventClassifier",
    "Captioner",
    "ClipEncoder",
    "Detection",
    "LazyModel",
    "ObjectDetector",
    "OcrReader",
    "OcrResult",
    "TextEncoder",
    "Transcriber",
    "Transcript",
    "TranscriptSegment",
    "ViolenceScore",
]
