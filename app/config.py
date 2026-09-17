"""Application configuration, loaded from environment variables (prefix ``APP_``)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings. Override any field via an ``APP_<NAME>`` env var or ``.env``."""

    model_config = SettingsConfigDict(env_prefix="APP_", env_file=".env", extra="ignore")

    # --- Storage ---
    data_dir: Path = Path("data")

    # --- Transcription (faster-whisper) ---
    whisper_model: str = "large-v3"
    device: str = "cuda"  # "cuda" or "cpu"
    compute_type: str = "float16"  # "float16" | "int8_float16" | "int8"
    language: str | None = None  # None -> automatic language detection
    beam_size: int = 5

    # --- Speaker diarization (token-free ECAPA embeddings) ---
    diarization_enabled: bool = True
    embedding_model: str = "speechbrain/spkrec-ecapa-voxceleb"
    window_seconds: float = 2.0
    hop_seconds: float = 1.0
    # Minimum cosine separation between the two coarsest voice groups to accept
    # more than one speaker (auto mode). Higher = more conservative (fewer
    # speakers, raise if one voice gets split); lower = splits more readily.
    speaker_separation: float = 0.65
    min_speakers: int = 1
    max_speakers: int = 10

    # --- Uploads ---
    max_upload_mb: int = 4096
    sample_rate: int = 16000  # Whisper/ECAPA both expect 16 kHz mono

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def outputs_dir(self) -> Path:
        return self.data_dir / "outputs"

    @property
    def jobs_dir(self) -> Path:
        # Legacy JSON job store; kept only so old jobs can be migrated into SQLite.
        return self.data_dir / "jobs"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "chedice.db"

    def ensure_dirs(self) -> None:
        for directory in (self.uploads_dir, self.outputs_dir):
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Return a cached, process-wide :class:`Settings` instance."""
    return Settings()
