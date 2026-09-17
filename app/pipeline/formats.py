"""Serialize a transcript into downloadable formats: SRT, VTT, TXT, JSON.

Each exporter accepts an optional ``names`` mapping that replaces the canonical
speaker labels ("SPEAKER 1", ...) with user-chosen names at render time.
"""

from __future__ import annotations

import json

from app.models import Job, Segment

EXTENSIONS = {"srt": "srt", "vtt": "vtt", "txt": "txt", "json": "json"}
MEDIA_TYPES = {
    "srt": "application/x-subrip; charset=utf-8",
    "vtt": "text/vtt; charset=utf-8",
    "txt": "text/plain; charset=utf-8",
    "json": "application/json; charset=utf-8",
}

Names = dict[str, str] | None


def _resolve(speaker: str | None, names: Names) -> str | None:
    if not speaker:
        return None
    return names.get(speaker, speaker) if names else speaker


def _timestamp(seconds: float, *, millis_sep: str) -> str:
    if seconds < 0:
        seconds = 0.0
    hours, rem = divmod(int(seconds), 3600)
    minutes, secs = divmod(rem, 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{millis_sep}{millis:03d}"


def _label(segment: Segment, names: Names) -> str:
    speaker = _resolve(segment.speaker, names)
    return f"{speaker}: " if speaker else ""


def to_srt(segments: list[Segment], names: Names = None) -> str:
    lines: list[str] = []
    for i, seg in enumerate(segments, start=1):
        start = _timestamp(seg.start, millis_sep=",")
        end = _timestamp(seg.end, millis_sep=",")
        lines.append(str(i))
        lines.append(f"{start} --> {end}")
        lines.append(f"{_label(seg, names)}{seg.text}")
        lines.append("")
    return "\n".join(lines)


def to_vtt(segments: list[Segment], names: Names = None) -> str:
    lines: list[str] = ["WEBVTT", ""]
    for seg in segments:
        start = _timestamp(seg.start, millis_sep=".")
        end = _timestamp(seg.end, millis_sep=".")
        lines.append(f"{start} --> {end}")
        lines.append(f"{_label(seg, names)}{seg.text}")
        lines.append("")
    return "\n".join(lines)


def to_txt(segments: list[Segment], names: Names = None) -> str:
    """Readable transcript, merging consecutive lines from the same speaker."""
    lines: list[str] = []
    current_speaker: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        if not buffer:
            return
        prefix = f"{current_speaker}: " if current_speaker else ""
        lines.append(f"{prefix}{' '.join(buffer)}")

    for seg in segments:
        speaker = _resolve(seg.speaker, names)
        if speaker != current_speaker:
            flush()
            buffer = []
            current_speaker = speaker
        buffer.append(seg.text)
    flush()

    return "\n\n".join(lines) + ("\n" if lines else "")


def to_json(job: Job, names: Names = None) -> str:
    payload = {
        "filename": job.filename,
        "language": job.detected_language,
        "duration": job.duration,
        "speaker_count": job.speaker_count,
        "segments": [
            {
                "start": seg.start,
                "end": seg.end,
                "text": seg.text,
                "speaker": _resolve(seg.speaker, names),
            }
            for seg in job.segments
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def render(fmt: str, job: Job) -> str:
    """Render ``job`` to ``fmt``, applying the job's custom speaker names."""
    names = job.speaker_names or None
    if fmt == "srt":
        return to_srt(job.segments, names)
    if fmt == "vtt":
        return to_vtt(job.segments, names)
    if fmt == "txt":
        return to_txt(job.segments, names)
    if fmt == "json":
        return to_json(job, names)
    raise ValueError(f"Unknown format: {fmt}")
