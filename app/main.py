"""FastAPI application: upload videos, track progress, download transcripts."""

from __future__ import annotations

import logging
import re
import subprocess
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app import llm, utils
from app.config import get_settings
from app.jobs import JobStore, Worker, new_job_id, now
from app.models import Job, Segment
from app.pipeline import audio, formats

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("chedice")

settings = get_settings()
store = JobStore(settings)
worker = Worker(store, settings)

STATIC_DIR = Path(__file__).parent / "static"
_UPLOAD_CHUNK = 1024 * 1024
_YOUTUBE_URL = re.compile(r"^https?://(www\.|m\.|music\.)?(youtube\.com/|youtu\.be/)", re.IGNORECASE)
_SAMPLE_MAX_SECONDS = 6.0  # longest voice sample cut for verification playback
_MAX_TRANSCRIPT_CHARS = 60000  # cap the transcript sent to the LLM as context


def _representative_segment(job: Job, speaker: str) -> Segment | None:
    """Pick the longest segment attributed to ``speaker`` (best voice sample)."""
    candidates = [s for s in job.segments if s.speaker == speaker and s.end > s.start]
    return max(candidates, key=lambda s: s.end - s.start, default=None)


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings.ensure_dirs()
    store.load()
    _separate_legacy_vision_settings()
    worker.start()
    worker.requeue_pending()
    logger.info("chedice ready — device=%s model=%s", settings.device, settings.whisper_model)
    yield
    worker.stop()


app = FastAPI(title="chedice", version="0.1.0", lifespan=lifespan)


def _separate_legacy_vision_settings() -> None:
    """Move the previous combined Qwen vision settings out of the text assistant."""
    raw = store.get_config()
    legacy_model = str(raw.get("llm_model") or "")
    if raw.get("vision_model") or "Qwen3-VL" not in legacy_model:
        return
    if not llm.is_vision_base_url(str(raw.get("llm_base_url") or "")):
        return
    store.set_config({
        "vision_enabled": raw.get("llm_enabled", "false"),
        "vision_base_url": raw.get("llm_base_url", ""),
        "vision_model": legacy_model,
        "vision_temperature": raw.get("llm_temperature", "0.3"),
        "llm_enabled": "false",
        "llm_base_url": "",
        "llm_model": "",
        "llm_api_key": "",
    })


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


def _normalize_category(category: object) -> str | None:
    """Trim a category label; blank selections become ``None``."""
    label = str(category).strip() if category not in (None, "") else ""
    return label or None


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse((STATIC_DIR / "index.html").read_text(encoding="utf-8"))


@app.post("/api/jobs")
async def create_job(
    file: UploadFile = File(...),
    num_speakers: int | None = Form(None),
    language: str | None = Form(None),
    category: str | None = Form(None),
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
        category=_normalize_category(category),
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
        category=_normalize_category(payload.get("category")),
    )
    store.add(job)
    worker.enqueue(job.id)
    return job.public_dict()


@app.get("/api/categories")
async def list_categories() -> list[str]:
    """Distinct category labels already used, for the group-by picker."""
    return store.categories()


@app.get("/api/config")
async def get_config() -> dict:
    """Return the current LLM connection settings for the configuration form."""
    config = llm.LLMConfig.from_store(store.get_config())
    public = config.to_public_dict()
    if public["enabled"] and not llm.is_text_base_url(config.base_url):
        public.update({"enabled": False, "base_url": "", "api_key": "", "model": ""})
    return public


@app.get("/api/vision-config")
async def get_vision_config() -> dict:
    """Return the independent vision-model settings."""
    config = llm.LLMConfig.from_store(store.get_config(), prefix="vision")
    return config.to_public_dict()


@app.put("/api/config")
async def update_config(payload: dict = Body(...)) -> dict:
    """Persist the LLM connection settings entered in the configuration mask."""
    base_url = str(payload.get("base_url") or "").strip()
    if payload.get("enabled") and base_url and not llm.is_text_base_url(base_url):
        raise HTTPException(
            status_code=422,
            detail="Configura un endpoint testuale locale, separato dal runtime vision.",
        )
    values = {
        "llm_enabled": "true" if payload.get("enabled") else "false",
        "llm_base_url": base_url,
        "llm_api_key": "",
        "llm_model": str(payload.get("model") or "").strip(),
        "llm_temperature": str(payload.get("temperature", 0.3)),
    }
    store.set_config(values)
    return llm.LLMConfig.from_store(store.get_config()).to_public_dict()


@app.put("/api/vision-config")
async def update_vision_config(payload: dict = Body(...)) -> dict:
    """Persist vision runtime settings independently from the text assistant."""
    base_url = str(payload.get("base_url") or "").strip()
    if payload.get("enabled") and base_url and not llm.is_vision_base_url(base_url):
        raise HTTPException(
            status_code=422,
            detail="Configura un endpoint locale per il modello vision.",
        )
    values = {
        "vision_enabled": "true" if payload.get("enabled") else "false",
        "vision_base_url": base_url,
        "vision_api_key": "",
        "vision_model": str(payload.get("model") or "").strip(),
        "vision_temperature": str(payload.get("temperature", 0.3)),
    }
    store.set_config(values)
    return llm.LLMConfig.from_store(store.get_config(), prefix="vision").to_public_dict()


@app.get("/api/jobs")
async def list_jobs() -> list[dict]:
    # Keep the list lightweight: omit the (potentially large) transcript.
    result = []
    for job in store.list():
        data = job.public_dict()
        data.pop("segments", None)
        result.append(data)
    return result


@app.get("/api/archive")
async def archive() -> list[dict]:
    """List completed transcriptions from the SQLite archive.

    Reads metadata straight from the database (no transcript segments) and adds
    a download URL per format so the archive can be browsed and exported.
    """
    items = []
    for row in store.archive():
        row["formats"] = {
            fmt: f"/api/jobs/{row['id']}/download/{fmt}" for fmt in formats.EXTENSIONS
        }
        items.append(row)
    return items


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


@app.get("/api/jobs/{job_id}/sample/{speaker}")
async def speaker_sample(job_id: str, speaker: str) -> Response:
    """Return a short audio clip of a representative moment for ``speaker``.

    Lets the user hear each detected voice and confirm the diarization is right.
    The clip is cut on demand from the stored source media with ffmpeg (so any
    input format plays in the browser) and cached on disk for repeat plays.
    """
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job non trovato.")

    segment = _representative_segment(job, speaker)
    if segment is None:
        raise HTTPException(status_code=404, detail="Parlante non trovato.")

    source = Path(job.source_path) if job.source_path else None
    if source is None or not source.exists():
        raise HTTPException(status_code=404, detail="Audio sorgente non più disponibile.")

    cache_path = settings.uploads_dir / f"{job_id}_{utils.safe_name(speaker)}.sample.mp3"
    if not cache_path.exists():
        duration = min(segment.end - segment.start, _SAMPLE_MAX_SECONDS)
        try:
            audio.extract_clip(source, cache_path, start=segment.start, duration=duration)
        except subprocess.CalledProcessError:
            cache_path.unlink(missing_ok=True)
            raise HTTPException(status_code=500, detail="Impossibile estrarre il campione audio.")

    return Response(
        content=cache_path.read_bytes(),
        media_type="audio/mpeg",
        headers={"Cache-Control": "no-cache"},
    )


@app.patch("/api/jobs/{job_id}/category")
async def set_category(job_id: str, payload: dict = Body(...)) -> dict:
    """Set (or clear) the grouping category of an existing job."""
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job non trovato.")
    job.category = _normalize_category(payload.get("category"))
    store.save(job)
    return job.public_dict()


@app.post("/api/jobs/{job_id}/ask")
async def ask_job(job_id: str, payload: dict = Body(...)) -> StreamingResponse:
    """Stream an AI answer about a completed transcript, token by token.

    The transcript (with any custom speaker names) is passed as context, so the
    user can ask e.g. "fammi un riassunto" or "cosa afferma Marco nel video?".
    The response is streamed as plain text to the browser as it is generated;
    once complete, the full exchange is appended to the job's AI history.
    """
    prompt = str(payload.get("prompt") or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Scrivi una domanda o una richiesta.")

    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job non trovato.")
    if not job.segments:
        raise HTTPException(status_code=409, detail="Trascrizione non ancora disponibile.")

    config = llm.LLMConfig.from_store(store.get_config())
    if not llm.is_text_base_url(config.base_url):
        raise HTTPException(
            status_code=409,
            detail="Configura un modello testuale locale distinto dal modello vision.",
        )
    if not config.is_ready:
        raise HTTPException(
            status_code=409,
            detail="Il modello LLM non è configurato. Aprilo dalle impostazioni ⚙️.",
        )

    transcript = formats.to_txt(job.segments, job.speaker_names or None)
    if len(transcript) > _MAX_TRANSCRIPT_CHARS:
        transcript = transcript[:_MAX_TRANSCRIPT_CHARS] + "\n\n[…trascrizione troncata…]"

    messages = [
        {
            "role": "system",
            "content": (
                "Sei un assistente che risponde a domande e produce riassunti a partire "
                "dalla trascrizione di un video/audio fornita qui sotto. Basati solo sul "
                "contenuto della trascrizione; se l'informazione non è presente, dillo "
                "chiaramente. Rispondi nella lingua della richiesta dell'utente.\n\n"
                f"=== TRASCRIZIONE ({job.filename}) ===\n{transcript}\n=== FINE TRASCRIZIONE ==="
            ),
        },
        {"role": "user", "content": prompt},
    ]

    # Open (and validate) the connection before streaming, so config/connection
    # errors become a proper HTTP error instead of a broken stream.
    try:
        resp = await run_in_threadpool(
            llm.open_stream, config, messages, read_timeout=settings.llm_request_timeout
        )
    except llm.LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    def generate():
        parts: list[str] = []
        try:
            for piece in llm.iter_stream(resp):
                parts.append(piece)
                yield piece
        finally:
            answer = "".join(parts).strip()
            if answer:  # persist only a non-empty exchange
                job.ai_messages.append(
                    {"prompt": prompt, "answer": answer, "created_at": now()}
                )
                store.save(job)

    return StreamingResponse(
        generate(),
        media_type="text/plain; charset=utf-8",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/jobs/{job_id}/describe-scenes")
async def describe_scenes(job_id: str, payload: dict = Body(default={})) -> dict:
    """Analyze sampled video frames with a local llama.cpp vision model."""
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job non trovato.")
    if job.status != "completed":
        raise HTTPException(status_code=409, detail="Attendi il completamento della trascrizione.")
    if not job.has_video or not job.source_path:
        raise HTTPException(status_code=409, detail="Questo job non contiene un video analizzabile.")
    if job.visual_status in ("queued", "processing"):
        raise HTTPException(status_code=409, detail="Analisi visiva già in corso.")
    config = llm.LLMConfig.from_store(store.get_config(), prefix="vision")
    if not config.is_ready:
        raise HTTPException(
            status_code=409,
            detail="Configura il modello vision nella sezione dedicata delle impostazioni ⚙️.",
        )
    if not llm.is_vision_base_url(config.base_url):
        raise HTTPException(status_code=409, detail="Configura un runtime vision locale.")
    detail = str(payload.get("detail", "medium")).lower()
    if detail not in {"base", "medium", "advanced"}:
        raise HTTPException(status_code=422, detail="Livello di dettaglio non valido.")
    job.visual_detail = detail
    job.visual_description = ""
    job.visual_status = "queued"
    job.visual_error = None
    store.save(job)
    worker.enqueue_scene_analysis(job.id)
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
