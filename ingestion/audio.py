"""Standalone audio file processing: normalize to 16kHz mono WAV."""
from __future__ import annotations

import subprocess
from pathlib import Path

from .config import Config
from .scanner import FileObject


def process_audio(file_obj: FileObject, cfg: Config) -> dict:
    stem = file_obj.path.stem
    out_dir = cfg.audio_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stem}_16k.wav"
    cmd = [
        "ffmpeg", "-y", "-i", str(file_obj.path),
        "-ac", "1", "-ar", str(cfg.audio_sample_rate_hz),
        str(out_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return {"audio_path": str(out_path)}
