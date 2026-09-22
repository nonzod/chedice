"""Audio extraction helpers built on ffmpeg/ffprobe."""

from __future__ import annotations

import subprocess
from pathlib import Path


def probe_duration(source: Path) -> float | None:
    """Return the media duration in seconds, or ``None`` if it cannot be read."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(source),
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError):
        return None


def has_video_stream(source: Path) -> bool:
    """Return whether media contains at least one video stream."""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(source),
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return bool(result.stdout.strip())
    except subprocess.CalledProcessError:
        return False


def extract_video_frames(
    source: Path, dest_dir: Path, duration: float | None
) -> list[tuple[float, Path]]:
    """Extract up to 8 resized representative frames, spread across the video."""
    if not has_video_stream(source):
        raise ValueError("Il file non contiene una traccia video.")
    if not duration or duration <= 0:
        duration = probe_duration(source)
    if not duration or duration <= 0:
        raise ValueError("Non riesco a determinare la durata del video.")

    count = min(8, max(1, int(duration // 15) + 1))
    times = [duration * (index + 0.5) / count for index in range(count)]
    dest_dir.mkdir(parents=True, exist_ok=True)
    frames: list[tuple[float, Path]] = []
    for index, timestamp in enumerate(times):
        target = dest_dir / f"frame-{index:02d}.jpg"
        cmd = [
            "ffmpeg", "-y", "-ss", f"{timestamp:.3f}", "-i", str(source),
            "-frames:v", "1", "-vf", "scale='min(640,iw)':-2",
            "-q:v", "5", str(target),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        if target.exists() and target.stat().st_size:
            frames.append((timestamp, target))
    if not frames:
        raise ValueError("Non è stato possibile estrarre fotogrammi dal video.")
    return frames


def extract_audio(source: Path, dest: Path, sample_rate: int = 16000) -> None:
    """Extract mono PCM audio from a media file at the given sample rate.

    Raises:
        subprocess.CalledProcessError: if ffmpeg fails (e.g. no audio stream).
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(source),
        "-vn",  # drop video
        "-ac",
        "1",  # mono
        "-ar",
        str(sample_rate),
        "-f",
        "wav",
        str(dest),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def extract_clip(source: Path, dest: Path, *, start: float, duration: float) -> None:
    """Cut a short MP3 clip ``[start, start + duration)`` for browser playback.

    Seeking before ``-i`` makes ffmpeg jump straight to the offset, so cutting a
    few seconds out of a long file stays fast regardless of its length.

    Raises:
        subprocess.CalledProcessError: if ffmpeg fails.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{max(0.0, start):.3f}",
        "-i",
        str(source),
        "-t",
        f"{max(0.1, duration):.3f}",
        "-vn",  # drop video
        "-ac",
        "1",  # mono
        "-c:a",
        "libmp3lame",
        "-q:a",
        "5",
        str(dest),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
