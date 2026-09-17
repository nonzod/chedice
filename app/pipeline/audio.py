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
