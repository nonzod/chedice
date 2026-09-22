"""Download video media from a URL (YouTube et al.) via yt-dlp."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import yt_dlp

logger = logging.getLogger(__name__)


class DownloadError(Exception):
    """Raised with a user-friendly message when a download cannot be completed."""


def _friendly_message(raw: str) -> str:
    text = raw.lower()
    if "requested format is not available" in text:
        return "YouTube non ha fornito un formato video/audio compatibile. Riprova più tardi."
    if "private" in text:
        return "Il video è privato."
    if "age" in text and "restrict" in text:
        return "Il video ha restrizioni di età e non può essere scaricato."
    if "not available in your country" in text or "geo" in text:
        return "Il video non è disponibile in questa area geografica."
    if "removed" in text or "no longer available" in text or "unavailable" in text:
        return "Il video non è più disponibile."
    return "Impossibile scaricare il video. Verifica che il link sia corretto."


def download_media(
    url: str,
    dest_dir: Path,
    job_id: str,
    progress_cb: Callable[[float], None] | None = None,
) -> tuple[Path, str]:
    """Download a muxed video with audio from ``url`` into ``dest_dir``.

    Args:
        url: source URL.
        dest_dir: directory to write the downloaded file into.
        job_id: used as the file name stem.
        progress_cb: optional callback receiving a 0..1 download fraction.

    Returns:
        A tuple ``(path, title)`` with the downloaded file and the video title.

    Raises:
        DownloadError: with a user-friendly message on any failure.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)

    def hook(status: dict) -> None:
        if progress_cb and status.get("status") == "downloading":
            total = status.get("total_bytes") or status.get("total_bytes_estimate")
            if total:
                progress_cb(min(status.get("downloaded_bytes", 0) / total, 1.0))

    final_path: Path | None = None

    def postprocessor_hook(status: dict) -> None:
        nonlocal final_path
        if status.get("status") == "finished":
            filepath = (status.get("info_dict") or {}).get("filepath")
            if filepath:
                final_path = Path(filepath)

    options = {
        "format": "bestvideo*+bestaudio/best",
        "outtmpl": str(dest_dir / f"{job_id}.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "progress_hooks": [hook],
        "postprocessor_hooks": [postprocessor_hook],
    }

    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            info = ydl.extract_info(url, download=True)
            if info.get("is_live"):
                raise DownloadError("I video in diretta (live) non sono supportati.")
            title = info.get("title") or "video"
            prepared_path = Path(ydl.prepare_filename(info))
            path = final_path if final_path and final_path.exists() else prepared_path
    except DownloadError:
        raise
    except yt_dlp.utils.DownloadError as exc:
        logger.warning("yt-dlp download failed for %s: %s", url, exc)
        raise DownloadError(_friendly_message(str(exc))) from exc
    except Exception as exc:  # noqa: BLE001 — surface anything unexpected cleanly
        logger.exception("Unexpected download error for %s", url)
        raise DownloadError("Errore imprevisto durante il download del video.") from exc

    if not path.exists():
        raise DownloadError("Il file video scaricato non è stato trovato.")

    return path, title
