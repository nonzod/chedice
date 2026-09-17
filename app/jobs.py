"""SQLite-backed job store with a single background worker.

Jobs (filename/URL, request options, transcript segments and speaker names) are
persisted in one SQLite database on the ``data`` volume, so transcripts survive
restarts. Jobs are cached in memory for fast reads; every mutation is written
through to SQLite. A single worker thread processes one job at a time, which
keeps GPU memory usage bounded (only one Whisper/ECAPA inference runs
concurrently).
"""

from __future__ import annotations

import json
import logging
import queue
import sqlite3
import threading
import time
from pathlib import Path

from app.config import Settings
from app.models import Job, JobStatus
from app.pipeline import Pipeline

logger = logging.getLogger(__name__)

# Scalar Job fields stored as their own (queryable) columns. Transcript segments
# and speaker names are stored as JSON text alongside them.
_SCALAR_COLUMNS = (
    "id",
    "filename",
    "stem",
    "source_path",
    "source_url",
    "created_at",
    "status",
    "progress",
    "stage",
    "requested_speakers",
    "requested_language",
    "detected_language",
    "speaker_count",
    "duration",
    "error",
)
_JSON_COLUMNS = ("speaker_names", "segments")
_ALL_COLUMNS = _SCALAR_COLUMNS + _JSON_COLUMNS

# Columns surfaced by the archive listing (no heavy transcript segments).
_ARCHIVE_COLUMNS = (
    "id",
    "filename",
    "stem",
    "source_url",
    "created_at",
    "detected_language",
    "duration",
    "speaker_count",
)

_UPSERT_SQL = (
    f"INSERT OR REPLACE INTO jobs ({','.join(_ALL_COLUMNS)}) "
    f"VALUES ({','.join(':' + c for c in _ALL_COLUMNS)})"
)

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    stem TEXT NOT NULL,
    source_path TEXT NOT NULL DEFAULT '',
    source_url TEXT,
    created_at REAL NOT NULL,
    status TEXT NOT NULL,
    progress REAL NOT NULL DEFAULT 0,
    stage TEXT,
    requested_speakers INTEGER,
    requested_language TEXT,
    detected_language TEXT,
    speaker_count INTEGER,
    duration REAL,
    error TEXT,
    speaker_names TEXT NOT NULL DEFAULT '{}',
    segments TEXT NOT NULL DEFAULT '[]'
)
"""


class JobStore:
    """Thread-safe registry of jobs persisted in a single SQLite database."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._jobs: dict[str, Job] = {}
        self._lock = threading.RLock()
        # The store is created at import time, before ``ensure_dirs`` runs, so
        # make sure the volume directory exists before opening the database.
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(
            settings.db_path, check_same_thread=False, isolation_level=None
        )
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute(_SCHEMA_SQL)

    # ---- Row <-> Job conversion ---------------------------------------
    @staticmethod
    def _row_params(job: Job) -> dict:
        data = job.to_dict()  # status already serialized to its string value
        data["speaker_names"] = json.dumps(job.speaker_names, ensure_ascii=False)
        data["segments"] = json.dumps([s.to_dict() for s in job.segments], ensure_ascii=False)
        return {col: data.get(col) for col in _ALL_COLUMNS}

    @staticmethod
    def _from_row(row: sqlite3.Row) -> Job:
        data = dict(row)
        data["speaker_names"] = json.loads(data.get("speaker_names") or "{}")
        data["segments"] = json.loads(data.get("segments") or "[]")
        return Job.from_dict(data)

    def _upsert(self, job: Job) -> None:
        self._db.execute(_UPSERT_SQL, self._row_params(job))

    # ---- Public API ----------------------------------------------------
    def load(self) -> None:
        """Load all jobs from SQLite, importing any legacy JSON store first."""
        with self._lock:
            self._migrate_legacy_json()
            for row in self._db.execute("SELECT * FROM jobs"):
                job = self._from_row(row)
                self._jobs[job.id] = job

    def save(self, job: Job) -> None:
        with self._lock:
            self._jobs[job.id] = job
            self._upsert(job)

    def add(self, job: Job) -> None:
        self.save(job)

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    def archive(self) -> list[dict]:
        """Completed transcriptions read straight from SQLite (metadata only).

        Projects just the columns needed to browse and download the archive, so
        the (potentially large) transcript segments are never loaded.
        """
        cols = ",".join(_ARCHIVE_COLUMNS)
        with self._lock:
            rows = self._db.execute(
                f"SELECT {cols} FROM jobs WHERE status = ? ORDER BY created_at DESC",
                (JobStatus.COMPLETED.value,),
            ).fetchall()
        return [dict(row) for row in rows]

    def delete(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.pop(job_id, None)
            if job is None:
                return False
            self._db.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
        if job.source_path:  # empty for a YouTube job not yet downloaded
            Path(job.source_path).unlink(missing_ok=True)
        for ext in ("srt", "vtt", "txt", "json"):
            (self._settings.outputs_dir / f"{job_id}.{ext}").unlink(missing_ok=True)
        return True

    # ---- One-time migration from the old JSON store --------------------
    def _migrate_legacy_json(self) -> None:
        legacy = self._settings.jobs_dir
        if not legacy.exists():
            return
        if self._db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]:
            return  # database already populated; nothing to migrate
        imported = 0
        for path in sorted(legacy.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                self._upsert(Job.from_dict(data))
                imported += 1
            except (json.JSONDecodeError, KeyError, OSError):
                logger.warning("Skipping corrupt legacy job file: %s", path)
        if imported:
            logger.info("Migrated %d legacy job(s) from JSON into SQLite", imported)


class Worker:
    """Background thread that drains a queue of job ids and processes them."""

    def __init__(self, store: JobStore, settings: Settings) -> None:
        self._store = store
        self._settings = settings
        self._queue: queue.Queue[str] = queue.Queue()
        self._pipeline: Pipeline | None = None
        self._thread = threading.Thread(target=self._run, name="chedice-worker", daemon=True)
        self._stop = threading.Event()

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._queue.put("")  # unblock the queue

    def enqueue(self, job_id: str) -> None:
        self._queue.put(job_id)

    def requeue_pending(self) -> None:
        """Re-queue jobs left unfinished by a previous run."""
        for job in self._store.list():
            if job.status in (JobStatus.QUEUED, JobStatus.PROCESSING):
                job.status = JobStatus.QUEUED
                job.progress = 0.0
                job.stage = "In coda"
                self._store.save(job)
                self.enqueue(job.id)

    def _run(self) -> None:
        while not self._stop.is_set():
            job_id = self._queue.get()
            if self._stop.is_set() or not job_id:
                break
            self._process(job_id)

    def _process(self, job_id: str) -> None:
        job = self._store.get(job_id)
        if job is None:
            return

        if self._pipeline is None:
            self._pipeline = Pipeline(self._settings)

        def on_update(progress: float, stage: str) -> None:
            job.progress = progress
            job.stage = stage
            self._store.save(job)

        job.status = JobStatus.PROCESSING
        job.stage = "Avvio"
        self._store.save(job)

        try:
            self._pipeline.run(job, on_update)
        except Exception as exc:  # noqa: BLE001 — surface any failure to the user
            logger.exception("Job %s failed", job_id)
            job.status = JobStatus.FAILED
            job.error = str(exc)
            job.stage = "Errore"
            self._store.save(job)


def new_job_id() -> str:
    from uuid import uuid4

    return uuid4().hex[:12]


def now() -> float:
    return time.time()
