"""Orchestrates the full processing pipeline for a single job.

Heavy models (Whisper, ECAPA) are loaded lazily on first use and then reused,
so the models live for the lifetime of the worker thread that owns the Pipeline.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from app import utils
from app.config import Settings
from app.models import Job, JobStatus
from app.pipeline import audio, download
from app.pipeline.diarize import Diarizer
from app.pipeline.transcribe import Transcriber

logger = logging.getLogger(__name__)

# Progress budget (in %) allotted to each stage.
_P_DOWNLOAD_END = 8.0
_P_EXTRACT = 10.0
_P_TRANSCRIBE_START = 12.0
_P_TRANSCRIBE_END = 70.0
_P_DIARIZE_END = 95.0

ProgressCb = Callable[[float, str], None]


class Pipeline:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._transcriber: Transcriber | None = None
        self._diarizer: Diarizer | None = None

    # ---- Lazy model loading -------------------------------------------
    @property
    def transcriber(self) -> Transcriber:
        if self._transcriber is None:
            logger.info("Loading Whisper model '%s'...", self.settings.whisper_model)
            self._transcriber = Transcriber(
                self.settings.whisper_model,
                device=self.settings.device,
                compute_type=self.settings.compute_type,
            )
        return self._transcriber

    @property
    def diarizer(self) -> Diarizer:
        if self._diarizer is None:
            logger.info("Loading speaker encoder '%s'...", self.settings.embedding_model)
            self._diarizer = Diarizer(
                self.settings.embedding_model,
                device=self.settings.device,
                cache_dir=str(self.settings.data_dir / "ecapa"),
            )
        return self._diarizer

    # ---- Main entry point ---------------------------------------------
    def run(self, job: Job, on_update: ProgressCb) -> None:
        """Process ``job`` end to end, mutating it and calling ``on_update``."""
        # 0) For YouTube jobs, download the audio first.
        if job.source_url and not job.source_path:
            self._download(job, on_update)

        source = Path(job.source_path)

        # 1) Extract audio.
        on_update(_P_EXTRACT, "Estrazione audio")
        job.duration = audio.probe_duration(source)
        wav_path = self.settings.uploads_dir / f"{job.id}.wav"
        audio.extract_audio(source, wav_path, sample_rate=self.settings.sample_rate)

        # 2) Transcribe.
        on_update(_P_TRANSCRIBE_START, "Trascrizione")

        def transcribe_progress(fraction: float) -> None:
            span = _P_TRANSCRIBE_END - _P_TRANSCRIBE_START
            on_update(_P_TRANSCRIBE_START + fraction * span, "Trascrizione")

        segments, language = self.transcriber.transcribe(
            wav_path,
            language=job.requested_language or self.settings.language,
            beam_size=self.settings.beam_size,
            duration=job.duration,
            progress_cb=transcribe_progress,
        )
        job.segments = segments
        job.detected_language = language
        on_update(_P_TRANSCRIBE_END, "Trascrizione completata")

        # 3) Diarize (optional).
        if self.settings.diarization_enabled and segments:
            on_update(_P_TRANSCRIBE_END, "Riconoscimento parlanti")
            try:
                job.speaker_count = self.diarizer.diarize(
                    wav_path,
                    segments,
                    window_seconds=self.settings.window_seconds,
                    hop_seconds=self.settings.hop_seconds,
                    speaker_separation=self.settings.speaker_separation,
                    num_speakers=job.requested_speakers,
                    min_speakers=self.settings.min_speakers,
                    max_speakers=self.settings.max_speakers,
                )
            except Exception:  # diarization must never lose a good transcript
                logger.exception("Diarization failed for job %s; keeping transcript", job.id)
                job.speaker_count = None
        on_update(_P_DIARIZE_END, "Finalizzazione")

        # Clean up the intermediate WAV (keep the original upload).
        wav_path.unlink(missing_ok=True)

        job.status = JobStatus.COMPLETED
        on_update(100.0, "Completato")

    def _download(self, job: Job, on_update: ProgressCb) -> None:
        """Download a YouTube job's audio and update its title/source path."""
        on_update(1.0, "Scaricamento video")
        last_pct = -1

        def download_progress(fraction: float) -> None:
            nonlocal last_pct
            pct = int(fraction * 100)
            if pct != last_pct:  # throttle disk writes to whole-percent changes
                last_pct = pct
                on_update(1.0 + fraction * (_P_DOWNLOAD_END - 1.0), "Scaricamento video")

        path, title = download.download_audio(
            job.source_url, self.settings.uploads_dir, job.id, progress_cb=download_progress
        )
        job.source_path = str(path)
        job.filename = title
        job.stem = utils.safe_stem(title)
        on_update(_P_DOWNLOAD_END, "Video scaricato")
