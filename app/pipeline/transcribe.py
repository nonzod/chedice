"""Speech-to-text using faster-whisper (CTranslate2)."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from faster_whisper import WhisperModel

from app.models import Segment

logger = logging.getLogger(__name__)


class Transcriber:
    """Wraps a loaded Whisper model. Instantiate once and reuse across jobs."""

    def __init__(self, model_name: str, device: str, compute_type: str) -> None:
        self.model = WhisperModel(model_name, device=device, compute_type=compute_type)

    def transcribe(
        self,
        audio_path: Path,
        *,
        language: str | None = None,
        beam_size: int = 5,
        duration: float | None = None,
        progress_cb: Callable[[float], None] | None = None,
    ) -> tuple[list[Segment], str]:
        """Transcribe an audio file.

        The VAD filter is applied first (it improves speech transcription), but if
        it removes *all* the audio — which happens with sung music or noisy tracks —
        we retry without it so we never return an empty transcript for real audio.

        Args:
            audio_path: 16 kHz mono WAV file.
            language: forced language code, or ``None`` for auto-detection.
            beam_size: decoder beam width.
            duration: media duration (seconds) used to estimate progress.
            progress_cb: called with a 0..1 fraction as segments are decoded.

        Returns:
            The list of segments and the (detected or forced) language code.
        """
        segments, detected = self._decode(
            audio_path,
            vad_filter=True,
            language=language,
            beam_size=beam_size,
            duration=duration,
            progress_cb=progress_cb,
        )
        if not segments:
            logger.info("VAD removed all audio for %s; retrying without VAD", audio_path.name)
            segments, detected = self._decode(
                audio_path,
                vad_filter=False,
                language=language,
                beam_size=beam_size,
                duration=duration,
                progress_cb=progress_cb,
            )
        return segments, detected

    def _decode(
        self,
        audio_path: Path,
        *,
        vad_filter: bool,
        language: str | None,
        beam_size: int,
        duration: float | None,
        progress_cb: Callable[[float], None] | None,
    ) -> tuple[list[Segment], str]:
        segment_iter, info = self.model.transcribe(
            str(audio_path),
            language=language,
            beam_size=beam_size,
            vad_filter=vad_filter,
        )

        segments: list[Segment] = []
        for seg in segment_iter:
            text = seg.text.strip()
            if not text:
                continue
            segments.append(Segment(start=seg.start, end=seg.end, text=text))
            if progress_cb and duration:
                progress_cb(min(seg.end / duration, 1.0))

        return segments, info.language
