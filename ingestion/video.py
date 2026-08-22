"""Video processing: audio extraction, scene segmentation, frame sampling."""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2

from .config import Config
from .scanner import FileObject


@dataclass
class Segment:
    index: int
    start_time: float
    end_time: float
    frame_paths: list[str]


def extract_audio_wav(video_path: Path, out_dir: Path, sample_rate_hz: int, stem: str) -> Path:
    """Extract audio to 16kHz mono WAV via ffmpeg."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stem}.wav"
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-ac", "1", "-ar", str(sample_rate_hz),
        "-vn", str(out_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return out_path


def _video_duration_sec(video_path: Path) -> float:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.release()
    if fps <= 0:
        return 0.0
    return frame_count / fps


def detect_scenes(video_path: Path, fallback_window_sec: int) -> list[tuple[float, float]]:
    """Return list of (start_sec, end_sec) using PySceneDetect content detection,
    falling back to fixed windows if no scenes are found or detection fails."""
    try:
        from scenedetect import open_video, SceneManager
        from scenedetect.detectors import ContentDetector

        video = open_video(str(video_path))
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector())
        scene_manager.detect_scenes(video=video)
        scene_list = scene_manager.get_scene_list()
        if scene_list:
            return [
                (start.get_seconds(), end.get_seconds()) for start, end in scene_list
            ]
    except Exception:
        pass

    duration = _video_duration_sec(video_path)
    if duration <= 0:
        return [(0.0, float(fallback_window_sec))]

    windows = []
    t = 0.0
    while t < duration:
        end = min(t + fallback_window_sec, duration)
        windows.append((t, end))
        t = end
    return windows


def extract_frames(
    video_path: Path,
    segments: list[tuple[float, float]],
    out_dir: Path,
    stem: str,
    frame_sample_rate_sec: int,
) -> list[Segment]:
    """Save one frame per frame_sample_rate_sec within each segment."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    results: list[Segment] = []
    for idx, (start, end) in enumerate(segments):
        frame_paths = []
        t = start
        while t < end:
            frame_index = int(t * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if ok:
                frame_path = out_dir / f"{stem}_seg{idx:04d}_t{t:07.2f}.jpg"
                cv2.imwrite(str(frame_path), frame)
                frame_paths.append(str(frame_path))
            t += frame_sample_rate_sec
        results.append(Segment(index=idx, start_time=start, end_time=end, frame_paths=frame_paths))

    cap.release()
    return results


def process_video(file_obj: FileObject, cfg: Config) -> dict:
    """Full video ingestion step: extract audio + segment + sample frames."""
    stem = file_obj.path.stem
    audio_path = extract_audio_wav(
        file_obj.path, cfg.audio_dir, cfg.audio_sample_rate_hz, stem
    )
    scenes = detect_scenes(file_obj.path, cfg.fallback_window_sec)
    segments = extract_frames(
        file_obj.path, scenes, cfg.frames_dir, stem, cfg.frame_sample_rate_sec
    )
    return {
        "audio_path": str(audio_path),
        "segments": segments,
    }
