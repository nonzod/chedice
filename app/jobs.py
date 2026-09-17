"""In-process job store with disk persistence and a single background worker.

A single worker thread processes one job at a time, which keeps GPU memory
usage bounded (only one Whisper/ECAPA inference runs concurrently).
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from pathlib import Path

from app.config import Settings
from app.models import Job, JobStatus
from app.pipeline import Pipeline

logger = logging.getLogger(__name__)


class JobStore:
    """Thread-safe registry of jobs, persisted as one JSON file per job."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def load_from_disk(self) -> None:
        for path in sorted(self._settings.jobs_dir.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                self._jobs[data["id"]] = Job.from_dict(data)
            except (json.JSONDecodeError, KeyError):
                logger.warning("Skipping corrupt job file: %s", path)

    def _path(self, job_id: str) -> Path:
        return self._settings.jobs_dir / f"{job_id}.json"

    def save(self, job: Job) -> None:
        with self._lock:
            self._jobs[job.id] = job
            tmp = self._path(job.id).with_suffix(".json.tmp")
            tmp.write_text(json.dumps(job.to_dict(), ensure_ascii=False), encoding="utf-8")
            tmp.replace(self._path(job.id))

    def add(self, job: Job) -> None:
        self.save(job)

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    def delete(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.pop(job_id, None)
            if job is None:
                return False
        self._path(job_id).unlink(missing_ok=True)
        if job.source_path:  # empty for a YouTube job not yet downloaded
            Path(job.source_path).unlink(missing_ok=True)
        for ext in ("srt", "vtt", "txt", "json"):
            (self._settings.outputs_dir / f"{job_id}.{ext}").unlink(missing_ok=True)
        return True


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
