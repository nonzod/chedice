"""Domain models: transcription segments and transcription jobs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum


class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Segment:
    """A single transcribed span of audio, optionally attributed to a speaker."""

    start: float
    end: float
    text: str
    speaker: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Segment:
        return cls(
            start=data["start"],
            end=data["end"],
            text=data["text"],
            speaker=data.get("speaker"),
        )


@dataclass
class Job:
    """A transcription job and its lifecycle state."""

    id: str
    filename: str  # original upload name (or video title for YouTube)
    stem: str  # base name used for downloads
    source_path: str  # stored media path (empty until a YouTube job is downloaded)
    created_at: float

    status: JobStatus = JobStatus.QUEUED
    progress: float = 0.0  # 0..100
    stage: str = "In coda"

    source_url: str | None = None  # set for YouTube jobs, downloaded by the worker

    requested_speakers: int | None = None  # None -> automatic
    requested_language: str | None = None  # None -> automatic

    detected_language: str | None = None
    speaker_count: int | None = None
    duration: float | None = None
    error: str | None = None

    # Optional custom labels: {"SPEAKER 1": "Marco", ...}. Applied on download.
    speaker_names: dict[str, str] = field(default_factory=dict)

    segments: list[Segment] = field(default_factory=list)

    # ---- Serialization -------------------------------------------------
    def to_dict(self) -> dict:
        data = asdict(self)
        data["status"] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: dict) -> Job:
        job = cls(
            id=data["id"],
            filename=data["filename"],
            stem=data["stem"],
            source_path=data["source_path"],
            created_at=data["created_at"],
            source_url=data.get("source_url"),
            status=JobStatus(data.get("status", "queued")),
            progress=data.get("progress", 0.0),
            stage=data.get("stage", "In coda"),
            requested_speakers=data.get("requested_speakers"),
            requested_language=data.get("requested_language"),
            detected_language=data.get("detected_language"),
            speaker_count=data.get("speaker_count"),
            duration=data.get("duration"),
            error=data.get("error"),
        )
        job.speaker_names = data.get("speaker_names", {})
        job.segments = [Segment.from_dict(s) for s in data.get("segments", [])]
        return job

    def public_dict(self) -> dict:
        """Job representation returned by the API (includes the transcript)."""
        return {
            "id": self.id,
            "filename": self.filename,
            "source_url": self.source_url,
            "status": self.status.value,
            "progress": round(self.progress, 1),
            "stage": self.stage,
            "requested_speakers": self.requested_speakers,
            "requested_language": self.requested_language,
            "detected_language": self.detected_language,
            "speaker_count": self.speaker_count,
            "duration": self.duration,
            "error": self.error,
            "created_at": self.created_at,
            "speaker_names": self.speaker_names,
            "segments": [s.to_dict() for s in self.segments],
        }
