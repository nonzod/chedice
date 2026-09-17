"""FastAPI application: upload videos, track progress, download transcripts."""

from __future__ import annotations

import logging
import re
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

from app import utils
from app.config import get_settings
from app.jobs import JobStore, Worker, new_job_id, now
from app.models import Job
from app.pipeline import formats

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("chedice")

settings = get_settings()
store = JobStore(settings)
worker = Worker(store, settings)

STATIC_DIR = Path(__file__).parent / "static"
_UPLOAD_CHUNK = 1024 * 1024
_YOUTUBE_URL = re.compile(r"^https?://(www\.|m\.|music\.)?(youtube\.com/|youtu\.be/)", re.IGNORECASE)


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings.ensure_dirs()
    store.load_from_disk()
    worker.start()
    worker.requeue_pending()
    logger.info("chedice ready — device=%s model=%s", settings.device, settings.whisper_model)
    yield
    worker.stop()


app = FastAPI(title="chedice", version="0.1.0", lifespan=lifespan)


def _normalize_options(num_speakers: object, language: object) -> tuple[int | None, str | None]:
    """Normalize 'auto'/blank UI selections (form strings or JSON) to ``None``."""
    try:
        speakers = int(num_speakers) if num_speakers not in (None, "") else None
    except (TypeError, ValueError):
        speakers = None
    if speakers is not None and speakers <= 0:
        speakers = None

    lang = str(language).strip() if language not in (None, "") else None
    if lang in (None, "auto"):
        lang = None
    return speakers, lang


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse((STATIC_DIR / "index.html").read_text(encoding="utf-8"))


@app.post("/api/jobs")
async def create_job(
    file: UploadFile = File(...),
    num_speakers: int | None = Form(None),
    language: str | None = Form(None),
) -> dict:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Nessun file fornito.")

    job_id = new_job_id()
    dest = settings.uploads_dir / f"{job_id}_{utils.safe_name(file.filename)}"

    size = 0
    max_bytes = settings.max_upload_mb * 1024 * 1024
    with dest.open("wb") as out:
        while chunk := await file.read(_UPLOAD_CHUNK):
            size += len(chunk)
            if size > max_bytes:
                out.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail="File troppo grande.")
            out.write(chunk)

    speakers, lang = _normalize_options(num_speakers, language)
    job = Job(
        id=job_id,
        filename=file.filename,
        stem=utils.safe_stem(file.filename),
        source_path=str(dest),
        created_at=now(),
        requested_speakers=speakers,
        requested_language=lang,
    )
    store.add(job)
    worker.enqueue(job.id)
    return job.public_dict()


@app.post("/api/jobs/youtube")
async def create_youtube_job(payload: dict = Body(...)) -> dict:
    """Create a job from a YouTube URL. The worker downloads the audio."""
    url = str(payload.get("url") or "").strip()
    if not _YOUTUBE_URL.match(url):
        raise HTTPException(status_code=400, detail="Inserisci un link YouTube valido.")

    speakers, lang = _normalize_options(payload.get("num_speakers"), payload.get("language"))
    job = Job(
        id=new_job_id(),
        filename="Video YouTube",  # replaced by the real title after download
        stem="youtube",
        source_path="",  # filled in by the download stage
        created_at=now(),
        source_url=url,
        requested_speakers=speakers,
        requested_language=lang,
    )
    store.add(job)
    worker.enqueue(job.id)
    return job.public_dict()


@app.get("/api/jobs")
async def list_jobs() -> list[dict]:
    # Keep the list lightweight: omit the (potentially large) transcript.
    result = []
    for job in store.list():
        data = job.public_dict()
        data.pop("segments", None)
        result.append(data)
    return result


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str) -> dict:
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job non trovato.")
    return job.public_dict()


@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str) -> dict:
    if not store.delete(job_id):
        raise HTTPException(status_code=404, detail="Job non trovato.")
    return {"deleted": job_id}


@app.patch("/api/jobs/{job_id}/speakers")
async def rename_speakers(job_id: str, names: dict[str, str] = Body(...)) -> dict:
    """Set custom speaker labels, e.g. ``{"SPEAKER 1": "Marco"}``.

    Only labels present in the transcript are accepted; blank or unchanged
    names are dropped. Applied to every download from now on.
    """
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job non trovato.")

    valid = {seg.speaker for seg in job.segments if seg.speaker}
    cleaned = {
        canonical: name.strip()
        for canonical, name in names.items()
        if canonical in valid and name and name.strip() and name.strip() != canonical
    }
    job.speaker_names = cleaned
    store.save(job)
    return job.public_dict()


@app.get("/api/jobs/{job_id}/download/{fmt}")
async def download(job_id: str, fmt: str) -> Response:
    if fmt not in formats.EXTENSIONS:
        raise HTTPException(status_code=400, detail="Formato non valido.")
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job non trovato.")
    if not job.segments:
        raise HTTPException(status_code=404, detail="Trascrizione non ancora disponibile.")

    ext = formats.EXTENSIONS[fmt]
    content = formats.render(fmt, job)
    return Response(
        content=content,
        media_type=formats.MEDIA_TYPES[fmt],
        headers={"Content-Disposition": f'attachment; filename="{job.stem}.{ext}"'},
    )


# Static assets (CSS/JS). Mounted last so it doesn't shadow the API routes.
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
