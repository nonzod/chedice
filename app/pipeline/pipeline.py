"""Orchestrates the full processing pipeline for a single job.

Heavy models (Whisper, ECAPA) are loaded lazily on first use and then reused,
so the models live for the lifetime of the worker thread that owns the Pipeline.
"""

from __future__ import annotations

import copy
import gc
import logging
from collections.abc import Callable
from pathlib import Path

from app import utils
from app.config import Settings
from app.models import Job, JobStatus, Segment
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


class JobCancelled(Exception):
    """Raised from a progress callback to abort a job the user cancelled."""


class Pipeline:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._transcriber: Transcriber | None = None
        self._transcriber_model: str | None = None
        self._diarizer: Diarizer | None = None

    def release_speech_models(self) -> None:
        """Release transcription/diarization allocations before local vision inference."""
        self._transcriber = None
        self._transcriber_model = None
        self._diarizer = None
        self._free_gpu()

    @staticmethod
    def _free_gpu() -> None:
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    # ---- Lazy model loading -------------------------------------------
    def transcriber(self, model_name: str) -> Transcriber:
        """Return a Transcriber for ``model_name``, swapping the loaded model if needed.

        Only one Whisper model is kept in memory at a time to bound VRAM usage.
        """
        if self._transcriber is None or self._transcriber_model != model_name:
            if self._transcriber is not None:
                logger.info("Unloading Whisper model '%s'", self._transcriber_model)
                self._transcriber = None
                self._free_gpu()
            logger.info("Loading Whisper model '%s'...", model_name)
            self._transcriber = Transcriber(
                model_name,
                device=self.settings.device,
                compute_type=self.settings.compute_type,
            )
            self._transcriber_model = model_name
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
        job.has_video = audio.has_video_stream(source)

        wav_path = self.settings.uploads_dir / f"{job.id}.wav"
        try:
            self._transcribe_and_diarize(job, source, wav_path, on_update)
        finally:
            # Clean up the intermediate WAV (keep the original upload), also when
            # the job fails or is cancelled.
            wav_path.unlink(missing_ok=True)

        job.status = JobStatus.COMPLETED
        on_update(100.0, "Completato")

    def _transcribe_and_diarize(
        self, job: Job, source: Path, wav_path: Path, on_update: ProgressCb
    ) -> None:
        # 1) Extract audio.
        on_update(_P_EXTRACT, "Estrazione audio")
        job.duration = audio.probe_duration(source)
        audio.extract_audio(source, wav_path, sample_rate=self.settings.sample_rate)

        # 2) Transcribe.
        on_update(_P_TRANSCRIBE_START, "Trascrizione")

        def transcribe_progress(fraction: float) -> None:
            span = _P_TRANSCRIBE_END - _P_TRANSCRIBE_START
            on_update(_P_TRANSCRIBE_START + fraction * span, "Trascrizione")

        model_name = job.requested_model or self.settings.whisper_model
        segments, language = self.transcriber(model_name).transcribe(
            wav_path,
            language=job.requested_language or self.settings.language,
            beam_size=self.settings.beam_size,
            duration=job.duration,
            progress_cb=transcribe_progress,
        )
        job.whisper_model = model_name
        job.segments = segments
        job.detected_language = language
        job.speaker_names = {}
        on_update(_P_TRANSCRIBE_END, "Trascrizione completata")

        # 3) Diarize (optional).
        if self.settings.diarization_enabled and segments:
            on_update(_P_TRANSCRIBE_END, "Riconoscimento parlanti")
            try:
                self._diarize(job, wav_path)
            except Exception:  # diarization must never lose a good transcript
                logger.exception("Diarization failed for job %s; keeping transcript", job.id)
                job.speaker_count = None
        on_update(_P_DIARIZE_END, "Finalizzazione")

    def rediarize(self, job: Job, on_update: ProgressCb) -> None:
        """Re-run speaker recognition on an existing transcript.

        Uses ``job.requested_speakers`` (``None`` = automatic). Custom speaker
        names are reset because the labels are reassigned from scratch.
        """
        source = Path(job.source_path) if job.source_path else None
        if source is None or not source.is_file():
            raise FileNotFoundError("Il file sorgente non è più disponibile.")
        on_update(20.0, "Estrazione audio")
        wav_path = self.settings.uploads_dir / f"{job.id}.wav"
        try:
            audio.extract_audio(source, wav_path, sample_rate=self.settings.sample_rate)
            on_update(40.0, "Riconoscimento parlanti")
            # Work on a copy: the transcript is replaced only if diarization succeeds
            # (and is not cancelled), so a failure never corrupts it.
            segments, count = self._run_diarizer(job, wav_path, copy.deepcopy(job.segments))
            on_update(95.0, "Finalizzazione")
        finally:
            wav_path.unlink(missing_ok=True)
        job.segments, job.speaker_count = segments, count
        job.speaker_names = {}

    def _diarize(self, job: Job, wav_path: Path) -> None:
        job.segments, job.speaker_count = self._run_diarizer(job, wav_path, job.segments)

    def _run_diarizer(
        self, job: Job, wav_path: Path, segments: list[Segment]
    ) -> tuple[list[Segment], int]:
        return self.diarizer.diarize(
            wav_path,
            segments,
            window_seconds=self.settings.window_seconds,
            hop_seconds=self.settings.hop_seconds,
            speaker_separation=self.settings.speaker_separation,
            num_speakers=job.requested_speakers,
            min_speakers=self.settings.min_speakers,
            max_speakers=self.settings.max_speakers,
            smoothing_radius=self.settings.smoothing_radius,
            min_turn_seconds=self.settings.min_turn_seconds,
        )

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

        path, title = download.download_media(
            job.source_url, self.settings.uploads_dir, job.id, progress_cb=download_progress
        )
        job.source_path = str(path)
        job.filename = title
        job.stem = utils.safe_stem(title)
        job.has_video = audio.has_video_stream(path)
        on_update(_P_DOWNLOAD_END, "Video scaricato")
