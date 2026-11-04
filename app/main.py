"""FastAPI application: upload videos, track progress, download transcripts."""

from __future__ import annotations

import logging
import re
import subprocess
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import quote

from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app import llm, utils
from app.config import get_settings
from app.jobs import JobStore, Worker, new_job_id, now
from app.models import Job, JobStatus, Segment
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
_SPEAKER_LABEL = re.compile(r"^SPEAKER \d+$")


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


def _normalize_model(model: object) -> str | None:
    """Validate a Whisper model chosen in the UI; blank/default -> ``None``."""
    name = str(model).strip() if model not in (None, "") else ""
    if not name or name in ("default", "auto"):
        return None
    if name not in settings.whisper_model_choices:
        raise HTTPException(status_code=422, detail=f"Modello Whisper non supportato: {name}")
    return None if name == settings.whisper_model else name


def _get_job(job_id: str) -> Job:
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job non trovato.")
    return job


def _require_editable(job: Job) -> None:
    """Transcript edits are allowed only on a finished, idle transcript."""
    if job.status != JobStatus.COMPLETED or not job.segments:
        raise HTTPException(status_code=409, detail="La trascrizione non è disponibile.")


def _after_speaker_change(job: Job) -> None:
    """Refresh derived speaker data after labels were edited, merged or reassigned."""
    present = set(job.speakers)
    job.speaker_count = len(present) or None
    job.speaker_names = {k: v for k, v in job.speaker_names.items() if k in present}
    store.clear_samples(job.id)


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
    whisper_model: str | None = Form(None),
) -> dict:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Nessun file fornito.")
    model = _normalize_model(whisper_model)

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
        requested_model=model,
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
    model = _normalize_model(payload.get("whisper_model"))
    job = Job(
        id=new_job_id(),
        filename="Video YouTube",  # replaced by the real title after download
        stem="youtube",
        source_path="",  # filled in by the download stage
        created_at=now(),
        source_url=url,
        requested_speakers=speakers,
        requested_language=lang,
        requested_model=model,
        category=_normalize_category(payload.get("category")),
    )
    store.add(job)
    worker.enqueue(job.id)
    return job.public_dict()


@app.get("/api/models")
async def list_models() -> dict:
    """Whisper models selectable from the UI, and the server default."""
    return {"default": settings.whisper_model, "choices": settings.whisper_model_choices}


@app.get("/api/categories")
async def list_categories() -> list[str]:
    """Distinct category labels already used, for the group-by picker."""
    return store.categories()


_ROLES = {
    # prefix: (URL validator, message when a URL is rejected)
    "llm": (
        llm.is_text_base_url,
        "L'assistente testuale non può usare il servizio vision http://llama:8080.",
    ),
    "vision": (
        llm.is_vision_base_url,
        "Il modello vision non può usare il servizio testuale.",
    ),
}


def _role_config(prefix: str) -> llm.LLMConfig:
    config = llm.LLMConfig.from_store(store.get_config(), prefix=prefix)
    valid, _ = _ROLES[prefix]
    # Ignore providers pointing at the other role's dedicated service.
    config.providers = [p for p in config.providers if valid(p.base_url)]
    return config


def _save_role(prefix: str, payload: dict) -> dict:
    """Validate and persist one role's provider chain from the settings form.

    Providers are saved in the given order, which defines their priority. A
    blank ``api_key`` keeps the key already stored for that provider id (the
    key is never sent back to the browser); ``clear_api_key`` removes it.
    """
    valid, message = _ROLES[prefix]
    current = llm.LLMConfig.from_store(store.get_config(), prefix=prefix)
    items = payload.get("providers")
    if not isinstance(items, list):
        raise HTTPException(status_code=422, detail="Elenco dei provider non valido.")

    providers: list[llm.Provider] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise HTTPException(status_code=422, detail="Provider non valido.")
        provider = llm.Provider.from_dict({**item, "priority": index + 1})
        if not provider.base_url and not provider.model:
            continue  # blank row in the form
        if not llm.is_valid_url(provider.base_url):
            raise HTTPException(
                status_code=422,
                detail=f"URL non valido per «{provider.label}»: usa http:// o https://.",
            )
        if not valid(provider.base_url):
            raise HTTPException(status_code=422, detail=message)
        if not provider.model:
            raise HTTPException(
                status_code=422, detail=f"Indica il modello per «{provider.label}»."
            )
        previous = current.find(provider.id)
        if not provider.api_key and previous and not item.get("clear_api_key"):
            provider.api_key = previous.api_key
        if provider.type == "ollama":
            provider.api_key = ""
        providers.append(provider)

    config = llm.LLMConfig(
        enabled=bool(payload.get("enabled")),
        fallback=bool(payload.get("fallback", True)),
        temperature=max(0.0, min(2.0, llm._parse_float(payload.get("temperature"), 0.3))),
        providers=providers,
    )
    if config.enabled and not config.chain:
        raise HTTPException(
            status_code=422, detail="Aggiungi almeno un provider attivo con URL e modello."
        )
    store.set_config(config.to_store_values(prefix))
    return _role_config(prefix).to_public_dict()


@app.get("/api/config")
async def get_config() -> dict:
    """Text-assistant settings (provider chain) for the configuration form."""
    return _role_config("llm").to_public_dict()


@app.get("/api/vision-config")
async def get_vision_config() -> dict:
    """Vision-model settings (provider chain), independent from the text assistant."""
    return _role_config("vision").to_public_dict()


@app.put("/api/config")
async def update_config(payload: dict = Body(...)) -> dict:
    """Persist the text-assistant provider chain entered in the settings mask."""
    return _save_role("llm", payload)


@app.put("/api/vision-config")
async def update_vision_config(payload: dict = Body(...)) -> dict:
    """Persist the vision provider chain independently from the text assistant."""
    return _save_role("vision", payload)


@app.post("/api/llm/test")
async def test_llm_provider(payload: dict = Body(...)) -> dict:
    """Check one provider from the settings form (reachability + model availability).

    A blank API key falls back to the key stored for that provider id, so a
    saved provider can be tested without re-typing it.
    """
    role = str(payload.get("role") or "llm")
    if role not in _ROLES:
        raise HTTPException(status_code=422, detail="Ruolo non valido.")
    item = payload.get("provider")
    if not isinstance(item, dict):
        raise HTTPException(status_code=422, detail="Provider non valido.")
    provider = llm.Provider.from_dict(item)
    if not provider.api_key:
        stored = llm.LLMConfig.from_store(store.get_config(), prefix=role).find(provider.id)
        if stored:
            provider.api_key = stored.api_key
    return await run_in_threadpool(llm.test_provider, provider)


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
    job = store.get(job_id)
    if job is not None and job.is_running:
        worker.cancel(job)  # stop the worker from processing a deleted job
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

    config = _role_config("llm")
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
    # Providers are tried in priority order until one accepts the request.
    try:
        stream = await run_in_threadpool(
            llm.open_stream, config, messages, read_timeout=settings.llm_request_timeout
        )
    except llm.LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    provider = stream.provider.label

    def generate():
        parts: list[str] = []
        try:
            for piece in llm.iter_stream(stream):
                parts.append(piece)
                yield piece
        finally:
            answer = "".join(parts).strip()
            if answer:  # persist only a non-empty exchange
                job.ai_messages.append(
                    {"prompt": prompt, "answer": answer, "created_at": now(),
                     "provider": provider}
                )
                store.save(job)

    return StreamingResponse(
        generate(),
        media_type="text/plain; charset=utf-8",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            # Percent-encoded: header values must be latin-1.
            "X-LLM-Provider": quote(provider),
        },
    )


@app.post("/api/jobs/{job_id}/describe-scenes")
async def describe_scenes(job_id: str, payload: dict = Body(default={})) -> dict:
    """Analyze sampled video frames with a local llama.cpp vision model."""
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job non trovato.")
    if job.status != "completed":
        raise HTTPException(status_code=409, detail="Attendi il completamento della trascrizione.")
    if not job.has_video:
        raise HTTPException(
            status_code=409, detail="Questo job non contiene un video analizzabile."
        )
    if not job.source_path or not Path(job.source_path).is_file():
        raise HTTPException(
            status_code=409, detail="Il file video originale non è più disponibile."
        )
    if job.visual_status in ("queued", "processing"):
        raise HTTPException(status_code=409, detail="Analisi visiva già in corso.")
    if not _role_config("vision").is_ready:
        raise HTTPException(
            status_code=409,
            detail="Configura il modello vision nella sezione dedicata delle impostazioni ⚙️.",
        )
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


@app.get("/api/jobs/{job_id}/details")
async def job_details(job_id: str) -> dict:
    """Everything stored about a job, for the "details" card in the UI."""
    job = _get_job(job_id)
    source = Path(job.source_path) if job.source_path else None
    media_available = bool(source and source.is_file())

    per_speaker: dict[str, dict] = {}
    words = 0
    for seg in job.segments:
        words += len(seg.text.split())
        if not seg.speaker:
            continue
        info = per_speaker.setdefault(
            seg.speaker,
            {"label": seg.speaker, "name": job.speaker_names.get(seg.speaker),
             "segments": 0, "seconds": 0.0},
        )
        info["segments"] += 1
        info["seconds"] += max(0.0, seg.end - seg.start)

    return {
        "id": job.id,
        "filename": job.filename,
        "status": job.status.value,
        "error": job.error,
        "created_at": job.created_at,
        "source": "youtube" if job.source_url else "upload",
        "source_url": job.source_url,
        "stored_file": source.name if source else None,
        "file_size": store.media_size(job) if media_available else None,
        "media_available": media_available,
        "has_video": job.has_video,
        "duration": job.duration,
        "requested_language": job.requested_language,
        "detected_language": job.detected_language,
        "whisper_model": job.whisper_model,
        "requested_model": job.requested_model or settings.whisper_model,
        "requested_speakers": job.requested_speakers,
        "speaker_count": job.speaker_count,
        "speakers": [
            {**info, "seconds": round(info["seconds"], 1)}
            for info in (per_speaker[label] for label in job.speakers)
        ],
        "segment_count": len(job.segments),
        "word_count": words,
        "category": job.category,
        "visual_status": job.visual_status,
        "visual_detail": job.visual_detail,
        "ai_message_count": len(job.ai_messages),
    }


@app.post("/api/jobs/{job_id}/cancel")
async def cancel_job(job_id: str) -> dict:
    """Cancel a queued or running job (a speaker re-recognition keeps the transcript)."""
    job = _get_job(job_id)
    if not job.is_running:
        raise HTTPException(status_code=409, detail="Il job non è in coda né in elaborazione.")
    worker.cancel(job)
    return job.public_dict()


@app.post("/api/jobs/{job_id}/retry")
async def retry_job(job_id: str, payload: dict = Body(default={})) -> dict:
    """Resume a cancelled or failed job, optionally with a different Whisper model."""
    job = _get_job(job_id)
    if job.status not in (JobStatus.CANCELLED, JobStatus.FAILED):
        raise HTTPException(status_code=409, detail="Solo i job annullati o falliti si riprendono.")
    has_source = bool(job.source_path and Path(job.source_path).is_file())
    if not has_source and not job.source_url:
        raise HTTPException(status_code=409, detail="Il file sorgente non è più disponibile.")
    if not has_source:
        job.source_path = ""  # YouTube job: download again
    if "whisper_model" in payload:
        job.requested_model = _normalize_model(payload.get("whisper_model"))
    worker.resubmit(job)
    return job.public_dict()


@app.post("/api/jobs/{job_id}/rediarize")
async def rediarize_job(job_id: str, payload: dict = Body(default={})) -> dict:
    """Re-run speaker recognition, e.g. with a forced number of speakers.

    Useful when two people were merged into one voice: the transcript text is
    kept, only the speaker attribution is recomputed (custom names are reset).
    """
    job = _get_job(job_id)
    _require_editable(job)
    if not job.source_path or not Path(job.source_path).is_file():
        raise HTTPException(status_code=409, detail="Il file sorgente non è più disponibile.")
    if job.visual_status in ("queued", "processing"):
        raise HTTPException(status_code=409, detail="Attendi la fine dell'analisi delle scene.")
    speakers, _ = _normalize_options(payload.get("num_speakers"), None)
    job.requested_speakers = speakers
    worker.resubmit(job, task="diarize")
    return job.public_dict()


@app.post("/api/jobs/{job_id}/speakers/merge")
async def merge_speakers(job_id: str, payload: dict = Body(...)) -> dict:
    """Merge voice ``source`` into ``target`` (same person detected as two voices)."""
    job = _get_job(job_id)
    _require_editable(job)
    source = str(payload.get("source") or "")
    target = str(payload.get("target") or "")
    present = set(job.speakers)
    if source not in present or target not in present or source == target:
        raise HTTPException(status_code=422, detail="Scegli due parlanti diversi esistenti.")
    for seg in job.segments:
        if seg.speaker == source:
            seg.speaker = target
    _after_speaker_change(job)
    store.save(job)
    return job.public_dict()


@app.patch("/api/jobs/{job_id}/segments")
async def edit_segments(job_id: str, payload: dict = Body(...)) -> dict:
    """Apply inline transcript edits.

    Body: ``{"edits": [{"index": 3, "text": "...", "speaker": "SPEAKER 2"}, ...]}``.
    ``text`` replaces the segment text (empty text deletes the segment);
    ``speaker`` reassigns it to an existing voice or to a new ``SPEAKER N`` label
    (two people detected as one voice).
    """
    job = _get_job(job_id)
    _require_editable(job)
    edits = payload.get("edits")
    if not isinstance(edits, list):
        raise HTTPException(status_code=422, detail="Formato delle modifiche non valido.")

    speakers_before = list(job.speakers)
    removed: set[int] = set()
    for edit in edits:
        if not isinstance(edit, dict):
            raise HTTPException(status_code=422, detail="Formato delle modifiche non valido.")
        try:
            index = int(edit.get("index"))
        except (TypeError, ValueError):
            raise HTTPException(status_code=422, detail="Indice di segmento non valido.") from None
        if not 0 <= index < len(job.segments):
            raise HTTPException(status_code=422, detail="Indice di segmento non valido.")
        seg = job.segments[index]
        if "text" in edit:
            text = " ".join(str(edit.get("text") or "").split())
            if not text:
                removed.add(index)
            elif text != seg.text:
                seg.text = text
                seg.words = []  # word timings no longer match the edited text
        if edit.get("speaker"):
            speaker = str(edit["speaker"]).strip()
            if speaker not in speakers_before and not _SPEAKER_LABEL.match(speaker):
                raise HTTPException(status_code=422, detail=f"Parlante non valido: {speaker}")
            seg.speaker = speaker

    if removed:
        job.segments = [s for i, s in enumerate(job.segments) if i not in removed]
    if job.speakers != speakers_before:
        _after_speaker_change(job)
    store.save(job)
    return job.public_dict()


@app.get("/api/retention")
async def get_retention() -> dict:
    """Retention policy plus current disk usage of the stored media."""
    return {**store.retention(), "uploads_bytes": store.uploads_usage()}


@app.put("/api/retention")
async def update_retention(payload: dict = Body(...)) -> dict:
    values = {}
    for key in ("media_days", "job_days"):
        try:
            days = int(payload.get(key) or 0)
        except (TypeError, ValueError):
            raise HTTPException(status_code=422, detail="Numero di giorni non valido.") from None
        if days < 0:
            raise HTTPException(status_code=422, detail="Numero di giorni non valido.")
        values[f"retention_{key}"] = str(days)
    store.set_config(values)
    return {**store.retention(), "uploads_bytes": store.uploads_usage()}


@app.post("/api/retention/cleanup")
async def run_cleanup() -> dict:
    """Apply the retention policy now."""
    result = await run_in_threadpool(store.cleanup)
    return {**result, **store.retention(), "uploads_bytes": store.uploads_usage()}


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
