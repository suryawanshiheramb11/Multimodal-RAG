"""End-to-end video tests against real encoded files.

These exercise the decode path for real rather than mocking it, because the
whole reason frame sampling moved from OpenCV seeking to a sequential PyAV pass
was decoder behaviour that only shows up on actual streams.
"""
from __future__ import annotations

import shutil
import subprocess

import pytest

from ingestion.media import AudioExtractor, FrameSampler, SceneSegmenter, probe_video
from ingestion.media.scenes import TimeSegment

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg is required to build fixtures"
)


#: lavfi source specs, with {d} standing in for the duration. Written out in
#: full because the separator before the first option differs per filter
#: (`testsrc=duration=…` but `color=c=gray:duration=…`).
TESTSRC = "testsrc=duration={d}:size=160x120:rate=10"
SOLID_GREY = "color=c=gray:duration={d}:size=160x120:rate=10"


def _encode(path, source_template: str, duration: int, with_audio: bool = True):
    command = ["ffmpeg", "-nostdin", "-loglevel", "error", "-y",
               "-f", "lavfi", "-i", source_template.format(d=duration)]
    if with_audio:
        command += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={duration}"]
    command += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-t", str(duration)]
    if with_audio:
        command += ["-c:a", "aac"]
    command += [str(path)]
    subprocess.run(command, check=True, capture_output=True)
    return path


@pytest.fixture(scope="module")
def plain_video(tmp_path_factory):
    return _encode(tmp_path_factory.mktemp("v") / "plain.mp4", TESTSRC, duration=6)


@pytest.fixture(scope="module")
def multi_scene_video(tmp_path_factory):
    """Three hard cuts between solid colours, which ContentDetector should see."""
    directory = tmp_path_factory.mktemp("scenes")
    parts = []
    for index, colour in enumerate(["red", "green", "blue"]):
        part = directory / f"{index}.mp4"
        subprocess.run(
            ["ffmpeg", "-nostdin", "-loglevel", "error", "-y",
             "-f", "lavfi", "-i", f"color=c={colour}:duration=2:size=160x120:rate=10",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", str(part)],
            check=True, capture_output=True,
        )
        parts.append(part)

    listing = directory / "parts.txt"
    listing.write_text("".join(f"file '{p}'\n" for p in parts))
    joined = directory / "scenes.mp4"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-f", "concat",
         "-safe", "0", "-i", str(listing), "-c", "copy", str(joined)],
        check=True, capture_output=True,
    )
    return joined


class TestProbe:
    def test_reports_duration_and_dimensions(self, plain_video):
        info = probe_video(plain_video)

        assert info.duration_sec == pytest.approx(6.0, abs=0.3)
        assert (info.width, info.height) == (160, 120)
        assert info.has_audio is True
        assert info.codec == "h264"


class TestFrameSampling:
    def test_one_frame_per_second_with_accurate_timestamps(self, plain_video, tmp_path):
        segments = [TimeSegment(index=0, start=0.0, end=5.0)]
        sampler = FrameSampler(interval_sec=1.0, max_frames=100)

        frames = sampler.sample(plain_video, segments, tmp_path)

        assert [round(f.timestamp) for f in frames[0]] == [0, 1, 2, 3, 4]
        # Each sample must land on (or just after) the requested second, never before.
        for expected, frame in zip(range(5), frames[0], strict=True):
            assert frame.timestamp >= expected - 0.001
            assert frame.timestamp < expected + 1.0
            assert frame.path.exists()

    def test_frames_are_grouped_by_segment(self, plain_video, tmp_path):
        segments = [
            TimeSegment(index=0, start=0.0, end=2.0),
            TimeSegment(index=1, start=2.0, end=4.0),
        ]

        frames = FrameSampler(interval_sec=1.0, max_frames=100).sample(
            plain_video, segments, tmp_path
        )

        assert set(frames) == {0, 1}
        assert len(frames[0]) == 2 and len(frames[1]) == 2
        assert all(f.timestamp < 2.0 for f in frames[0])
        assert all(f.timestamp >= 2.0 for f in frames[1])

    def test_global_frame_budget_is_enforced(self, plain_video, tmp_path):
        """Many short segments must not multiply past the per-file cap."""
        segments = [TimeSegment(index=i, start=float(i), end=float(i) + 1) for i in range(6)]

        frames = FrameSampler(interval_sec=0.2, max_frames=5).sample(
            plain_video, segments, tmp_path
        )

        assert sum(len(v) for v in frames.values()) == 5

    def test_written_frames_are_owner_only(self, plain_video, tmp_path):
        frames = FrameSampler(interval_sec=1.0, max_frames=2).sample(
            plain_video, [TimeSegment(0, 0.0, 2.0)], tmp_path
        )
        mode = frames[0][0].path.stat().st_mode & 0o777
        assert mode == 0o600


class TestSceneSegmentation:
    def test_hard_cuts_are_detected(self, multi_scene_video):
        segmenter = SceneSegmenter(threshold=27.0, fallback_window_sec=5.0,
                                   max_duration_sec=600)

        segments, strategy = segmenter.segment(multi_scene_video, duration_sec=6.0)

        assert strategy == "scene"
        assert len(segments) >= 2
        # Segments must tile the timeline without gaps.
        for earlier, later in zip(segments, segments[1:], strict=False):
            assert later.start == pytest.approx(earlier.end, abs=0.15)

    def test_static_video_falls_back_to_fixed_windows(self, tmp_path):
        static = _encode(tmp_path / "static.mp4", SOLID_GREY, duration=6, with_audio=False)
        segmenter = SceneSegmenter(threshold=27.0, fallback_window_sec=5.0,
                                   max_duration_sec=600)

        segments, strategy = segmenter.segment(static, duration_sec=6.0)

        assert strategy == "fixed"
        assert [(s.start, s.end) for s in segments] == [(0.0, 5.0), (5.0, 6.0)]


class TestAudioExtraction:
    def test_produces_16k_mono_wav(self, plain_video, tmp_path):
        import wave

        destination = tmp_path / "out.wav"
        AudioExtractor(sample_rate_hz=16000, timeout_sec=60).extract(plain_video, destination)

        with wave.open(str(destination), "rb") as handle:
            assert handle.getframerate() == 16000
            assert handle.getnchannels() == 1
            assert handle.getsampwidth() == 2
        assert destination.stat().st_mode & 0o777 == 0o600

    def test_silent_video_reports_no_audio_track(self, tmp_path):
        from ingestion.errors import MediaProcessingError

        silent = _encode(tmp_path / "silent.mp4", TESTSRC, duration=2, with_audio=False)

        with pytest.raises(MediaProcessingError, match="no audio track|ffmpeg failed"):
            AudioExtractor(sample_rate_hz=16000, timeout_sec=60).extract(
                silent, tmp_path / "silent.wav"
            )
