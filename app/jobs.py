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
from app.pipeline import JobCancelled, Pipeline

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
    "requested_model",
    "whisper_model",
    "task",
    "detected_language",
    "speaker_count",
    "duration",
    "error",
    "category",
    "has_video",
    "visual_description",
    "visual_detail",
    "visual_status",
    "visual_error",
)
_JSON_COLUMNS = ("speaker_names", "segments", "ai_messages")
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
    "category",
    "source_path",
    "has_video",
    "whisper_model",
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
    requested_model TEXT,
    whisper_model TEXT,
    task TEXT NOT NULL DEFAULT 'transcribe',
    detected_language TEXT,
    speaker_count INTEGER,
    duration REAL,
    error TEXT,
    category TEXT,
    has_video INTEGER NOT NULL DEFAULT 0,
    visual_description TEXT NOT NULL DEFAULT '',
    visual_detail TEXT NOT NULL DEFAULT 'medium',
    visual_status TEXT NOT NULL DEFAULT 'not_started',
    visual_error TEXT,
    speaker_names TEXT NOT NULL DEFAULT '{}',
    segments TEXT NOT NULL DEFAULT '[]',
    ai_messages TEXT NOT NULL DEFAULT '[]'
)
"""

# Key/value store for runtime app configuration (e.g. the LLM connection).
_SETTINGS_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS app_settings (
    key TEXT PRIMARY KEY,
    value TEXT
)
"""


class JobStore:
    """Thread-safe registry of jobs persisted in a single SQLite database."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._jobs: dict[str, Job] = {}
        self._deleted: set[str] = set()  # ids deleted while possibly still in use
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
        self._db.execute(_SETTINGS_SCHEMA_SQL)
        self._migrate_schema()

    def _migrate_schema(self) -> None:
        """Add columns introduced after a database was first created."""
        existing = {row["name"] for row in self._db.execute("PRAGMA table_info(jobs)")}
        if "category" not in existing:
            self._db.execute("ALTER TABLE jobs ADD COLUMN category TEXT")
        if "ai_messages" not in existing:
            self._db.execute(
                "ALTER TABLE jobs ADD COLUMN ai_messages TEXT NOT NULL DEFAULT '[]'"
            )
        for name, definition in (
            ("has_video", "INTEGER NOT NULL DEFAULT 0"),
            ("visual_description", "TEXT NOT NULL DEFAULT ''"),
            ("visual_detail", "TEXT NOT NULL DEFAULT 'medium'"),
            ("visual_status", "TEXT NOT NULL DEFAULT 'not_started'"),
            ("visual_error", "TEXT"),
            ("requested_model", "TEXT"),
            ("whisper_model", "TEXT"),
            ("task", "TEXT NOT NULL DEFAULT 'transcribe'"),
        ):
            if name not in existing:
                self._db.execute(f"ALTER TABLE jobs ADD COLUMN {name} {definition}")

    # ---- Row <-> Job conversion ---------------------------------------
    @staticmethod
    def _row_params(job: Job) -> dict:
        data = job.to_dict()  # status already serialized to its string value
        data["speaker_names"] = json.dumps(job.speaker_names, ensure_ascii=False)
        data["segments"] = json.dumps(
            [s.to_storage() for s in job.segments], ensure_ascii=False, separators=(",", ":")
        )
        data["ai_messages"] = json.dumps(job.ai_messages, ensure_ascii=False)
        data["has_video"] = int(job.has_video)
        return {col: data.get(col) for col in _ALL_COLUMNS}

    @staticmethod
    def _from_row(row: sqlite3.Row) -> Job:
        data = dict(row)
        data["speaker_names"] = json.loads(data.get("speaker_names") or "{}")
        data["segments"] = json.loads(data.get("segments") or "[]")
        data["ai_messages"] = json.loads(data.get("ai_messages") or "[]")
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
            if job.id in self._deleted:
                return  # a worker finishing a deleted job must not resurrect it
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
        items = []
        for row in rows:
            item = dict(row)
            source = item.pop("source_path", "") or ""
            item["has_video"] = bool(item.get("has_video"))
            item["media_available"] = bool(source) and Path(source).is_file()
            items.append(item)
        return items

    def categories(self) -> list[str]:
        """Distinct, non-empty category labels already in use, alphabetically."""
        with self._lock:
            rows = self._db.execute(
                "SELECT DISTINCT category FROM jobs "
                "WHERE category IS NOT NULL AND category != '' "
                "ORDER BY category COLLATE NOCASE"
            ).fetchall()
        return [row["category"] for row in rows]

    # ---- Runtime app configuration (key/value) -------------------------
    def get_config(self) -> dict[str, str]:
        """Return all persisted app settings as a plain ``{key: value}`` dict."""
        with self._lock:
            rows = self._db.execute("SELECT key, value FROM app_settings").fetchall()
        return {row["key"]: row["value"] for row in rows}

    def set_config(self, values: dict[str, str]) -> None:
        """Upsert the given settings; ``None`` values are stored as empty strings."""
        with self._lock:
            self._db.executemany(
                "INSERT OR REPLACE INTO app_settings (key, value) VALUES (?, ?)",
                [(key, "" if value is None else str(value)) for key, value in values.items()],
            )

    def delete(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.pop(job_id, None)
            if job is None:
                return False
            self._deleted.add(job_id)
            self._db.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
        self._delete_media(job)
        for ext in ("srt", "vtt", "txt", "json"):
            (self._settings.outputs_dir / f"{job_id}.{ext}").unlink(missing_ok=True)
        return True

    def clear_samples(self, job_id: str) -> None:
        """Drop cached voice-verification clips (stale after speaker changes)."""
        for sample in self._settings.uploads_dir.glob(f"{job_id}_*.sample.mp3"):
            sample.unlink(missing_ok=True)

    def _delete_media(self, job: Job) -> int:
        """Delete the job's stored media, samples and leftovers; return freed bytes."""
        uploads = self._settings.uploads_dir
        paths = set(uploads.glob(f"{job.id}_*")) | set(uploads.glob(f"{job.id}.*"))
        if job.source_path:  # empty for a YouTube job not yet downloaded
            paths.add(Path(job.source_path))
        freed = 0
        for path in paths:
            try:
                if path.is_file():
                    freed += path.stat().st_size
                    path.unlink()
            except OSError:
                logger.warning("Could not delete %s", path)
        return freed

    def media_size(self, job: Job) -> int | None:
        """Size in bytes of the stored source media, or ``None`` if not available."""
        if not job.source_path:
            return None
        try:
            return Path(job.source_path).stat().st_size
        except OSError:
            return None

    def uploads_usage(self) -> int:
        """Total bytes used by the uploads directory."""
        total = 0
        for path in self._settings.uploads_dir.glob("*"):
            try:
                if path.is_file():
                    total += path.stat().st_size
            except OSError:
                pass
        return total

    # ---- Retention -----------------------------------------------------
    def retention(self) -> dict[str, int]:
        """Current retention policy (days, 0 = never), env defaults overridden by the UI."""
        raw = self.get_config()

        def days(key: str, default: int) -> int:
            try:
                return max(0, int(raw.get(key, default)))
            except (TypeError, ValueError):
                return default

        return {
            "media_days": days("retention_media_days", self._settings.media_retention_days),
            "job_days": days("retention_job_days", self._settings.job_retention_days),
        }

    def cleanup(self) -> dict[str, int]:
        """Apply the retention policy to finished jobs.

        Jobs older than ``job_days`` are deleted entirely; for jobs older than
        ``media_days`` only the original media is removed (the transcript stays).
        Queued/processing jobs are never touched.
        """
        policy = self.retention()
        current = time.time()
        deleted_jobs = deleted_media = freed = 0
        for job in self.list():
            if job.is_running or job.visual_status in ("queued", "processing"):
                continue
            age_days = (current - job.created_at) / 86400
            if policy["job_days"] and age_days >= policy["job_days"]:
                freed += self.media_size(job) or 0
                if self.delete(job.id):
                    deleted_jobs += 1
                continue
            if policy["media_days"] and age_days >= policy["media_days"] and job.source_path:
                freed += self._delete_media(job)
                with self._lock:
                    job.source_path = ""
                    self.save(job)
                deleted_media += 1
        if deleted_jobs or deleted_media:
            logger.info(
                "Retention cleanup: %d job(s) deleted, media removed for %d job(s), %d MB freed",
                deleted_jobs, deleted_media, freed // (1024 * 1024),
            )
        return {"deleted_jobs": deleted_jobs, "deleted_media": deleted_media, "freed_bytes": freed}

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

    _CLEANUP_INTERVAL = 3600.0  # seconds between automatic retention passes

    def __init__(self, store: JobStore, settings: Settings) -> None:
        self._store = store
        self._settings = settings
        self._queue: queue.Queue[tuple[str, str]] = queue.Queue()
        self._pipeline: Pipeline | None = None
        self._thread = threading.Thread(target=self._run, name="chedice-worker", daemon=True)
        self._janitor = threading.Thread(target=self._janitor_loop, name="chedice-janitor",
                                         daemon=True)
        self._stop = threading.Event()
        self._cancel_lock = threading.Lock()
        self._cancel_requested: set[str] = set()
        self._current: str | None = None  # id of the job being processed

    def start(self) -> None:
        self._thread.start()
        self._janitor.start()

    def stop(self) -> None:
        self._stop.set()
        self._queue.put(("stop", ""))  # unblock the queue

    def enqueue(self, job_id: str) -> None:
        self._queue.put(("job", job_id))

    def enqueue_scene_analysis(self, job_id: str) -> None:
        self._queue.put(("scenes", job_id))

    # ---- Cancel / resume ----------------------------------------------
    def cancel(self, job: Job) -> None:
        """Cancel a queued or running job.

        A queued job is cancelled immediately (the worker skips it when it is
        dequeued). A running job is flagged and aborted at its next progress
        update. Cancelling a speaker re-recognition keeps the existing transcript.
        """
        with self._cancel_lock:
            if job.status == JobStatus.PROCESSING and self._current == job.id:
                self._cancel_requested.add(job.id)
                job.stage = "Annullamento…"
                self._store.save(job)
                return
        self._mark_cancelled(job)

    def _mark_cancelled(self, job: Job) -> None:
        if job.task == "diarize":
            job.status = JobStatus.COMPLETED  # transcript is untouched
            job.stage = "Completato"
            job.progress = 100.0
        else:
            job.status = JobStatus.CANCELLED
            job.stage = "Annullato"
        job.task = "transcribe"
        self._store.save(job)

    def _check_cancel(self, job_id: str) -> None:
        with self._cancel_lock:
            if job_id in self._cancel_requested:
                raise JobCancelled()

    def resubmit(self, job: Job, task: str = "transcribe") -> None:
        """(Re)queue ``job`` for ``task`` — used to resume or retry it."""
        job.task = task
        job.status = JobStatus.QUEUED
        job.progress = 0.0
        job.stage = "In coda"
        job.error = None
        self._store.save(job)
        self.enqueue(job.id)

    def requeue_pending(self) -> None:
        """Re-queue jobs left unfinished by a previous run."""
        for job in self._store.list():
            if job.status in (JobStatus.QUEUED, JobStatus.PROCESSING):
                self.resubmit(job, job.task)
            elif job.visual_status in ("queued", "processing"):
                job.visual_status = "queued"
                self._store.save(job)
                self.enqueue_scene_analysis(job.id)

    # ---- Loop ----------------------------------------------------------
    def _run(self) -> None:
        while not self._stop.is_set():
            task, job_id = self._queue.get()
            if self._stop.is_set() or task == "stop":
                break
            if task == "job":
                self._process(job_id)
            elif task == "scenes":
                self._analyze_scenes(job_id)

    def _janitor_loop(self) -> None:
        delay = 30.0  # first pass shortly after startup, then hourly
        while not self._stop.wait(timeout=delay):
            delay = self._CLEANUP_INTERVAL
            try:
                self._store.cleanup()
            except Exception:  # noqa: BLE001 — never kill the janitor thread
                logger.exception("Retention cleanup failed")

    def _process(self, job_id: str) -> None:
        job = self._store.get(job_id)
        # Skip jobs cancelled (or already handled) while they waited in the queue.
        if job is None or job.status != JobStatus.QUEUED:
            return

        if self._pipeline is None:
            self._pipeline = Pipeline(self._settings)

        def on_update(progress: float, stage: str) -> None:
            self._check_cancel(job.id)
            job.progress = progress
            job.stage = stage
            self._store.save(job)

        with self._cancel_lock:
            self._current = job.id
            self._cancel_requested.discard(job.id)
        job.status = JobStatus.PROCESSING
        job.stage = "Avvio"
        self._store.save(job)

        try:
            if job.task == "diarize":
                self._pipeline.rediarize(job, on_update)
                self._check_cancel(job.id)
                job.task = "transcribe"
                job.status = JobStatus.COMPLETED
                job.progress = 100.0
                job.stage = "Completato"
                self._store.clear_samples(job.id)
                self._store.save(job)
            else:
                self._pipeline.run(job, on_update)
                self._store.clear_samples(job.id)
        except Exception as exc:  # noqa: BLE001 — surface any failure to the user
            with self._cancel_lock:
                cancelled = job.id in self._cancel_requested
            if cancelled or isinstance(exc, JobCancelled):
                logger.info("Job %s cancelled", job_id)
                self._mark_cancelled(job)
            elif job.task == "diarize":
                logger.exception("Speaker re-recognition failed for job %s", job_id)
                job.task = "transcribe"
                job.status = JobStatus.COMPLETED  # the transcript is still valid
                job.progress = 100.0
                job.stage = "Completato"
                job.error = f"Riconoscimento parlanti non riuscito: {exc}"
                self._store.save(job)
            else:
                logger.exception("Job %s failed", job_id)
                job.status = JobStatus.FAILED
                job.error = str(exc)
                job.stage = "Errore"
                self._store.save(job)
        finally:
            with self._cancel_lock:
                self._current = None
                self._cancel_requested.discard(job.id)

    def _analyze_scenes(self, job_id: str) -> None:
        job = self._store.get(job_id)
        if job is None:
            return
        if not job.has_video or not job.source_path or not Path(job.source_path).is_file():
            job.visual_status = "failed"
            job.visual_error = "Il file video sorgente non è disponibile."
            self._store.save(job)
            return
        if self._pipeline is not None:
            self._pipeline.release_speech_models()

        from app import llm, vision

        job.visual_status = "processing"
        job.visual_error = None
        self._store.save(job)
        try:
            config = llm.LLMConfig.from_store(self._store.get_config(), prefix="vision")
            job.visual_description = vision.describe_video(
                Path(job.source_path),
                job.duration,
                config,
                self._settings.llm_request_timeout,
                job.visual_detail,
            )
            job.visual_status = "completed"
        except Exception as exc:  # noqa: BLE001 — preserve queue and surface the failure
            logger.exception("Scene analysis failed for job %s", job_id)
            if isinstance(exc, llm.LLMIncompleteError):
                job.visual_description = exc.partial_text
            job.visual_status = "failed"
            job.visual_error = f"Analisi incompleta: {exc}" if isinstance(exc, llm.LLMIncompleteError) else str(exc)
        self._store.save(job)


def new_job_id() -> str:
    from uuid import uuid4

    return uuid4().hex[:12]


def now() -> float:
    return time.time()
