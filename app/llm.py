"""Minimal client for OpenAI-compatible chat APIs (OpenRouter, Ollama, …).

The LLM connection is configured at runtime from the settings UI and persisted
in SQLite (see :class:`app.jobs.JobStore`), so no environment variables or
restarts are needed to point the app at a different endpoint or model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from urllib.parse import urlparse

import requests

# Keys used to persist the LLM configuration in the ``app_settings`` table.
CONFIG_KEYS = ("llm_enabled", "llm_base_url", "llm_api_key", "llm_model", "llm_temperature")

_DEFAULTS = {
    "llm_enabled": "false",
    "llm_base_url": "",
    "llm_api_key": "",
    "llm_model": "",
    "llm_temperature": "0.3",
}

# Connection timeout (fast fail if the host is unreachable) and the default
# read timeout, applied *between streamed chunks* — not to the whole generation.
# Streaming keeps data flowing, so a slow model no longer trips a single long
# read timeout; only a genuine stall (e.g. a very slow cold-start load) can.
_CONNECT_TIMEOUT = 10  # seconds
_DEFAULT_READ_TIMEOUT = 300  # seconds between chunks (survives model cold-start)


class LLMError(RuntimeError):
    """Raised when the LLM is not configured or the remote call fails."""


@dataclass
class LLMConfig:
    """Runtime LLM connection settings, as stored in SQLite."""

    enabled: bool = False
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    temperature: float = 0.3

    @classmethod
    def from_store(cls, raw: dict[str, str]) -> LLMConfig:
        data = {**_DEFAULTS, **{k: v for k, v in raw.items() if v is not None}}
        return cls(
            enabled=str(data["llm_enabled"]).lower() in ("1", "true", "yes", "on"),
            base_url=str(data["llm_base_url"]).strip(),
            api_key=str(data["llm_api_key"]).strip(),
            model=str(data["llm_model"]).strip(),
            temperature=_parse_float(data["llm_temperature"], 0.3),
        )

    def to_public_dict(self) -> dict:
        """Config for the settings form. Key is echoed (local, single-user app)."""
        return {
            "enabled": self.enabled,
            "base_url": self.base_url,
            "api_key": self.api_key,
            "model": self.model,
            "temperature": self.temperature,
        }

    @property
    def is_ready(self) -> bool:
        return bool(self.enabled and self.base_url and self.model)


def _parse_float(value: object, fallback: float) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return fallback


def _endpoint(base_url: str) -> str:
    """Build the chat-completions URL from a user-entered base URL.

    Tolerant of the common ways people enter it:
    - full endpoint (``…/chat/completions``) → used as-is;
    - bare host (``http://host:11434``, no path) → assume an OpenAI-compatible
      server mounted under ``/v1`` (Ollama, LocalAI, …);
    - anything else (``…/v1``, ``…/api/v1``) → append ``/chat/completions``.
    """
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    path = urlparse(base).path
    if path in ("", "/"):  # bare host:port → OpenAI-compatible APIs live under /v1
        return f"{base}/v1/chat/completions"
    return f"{base}/chat/completions"


def open_stream(
    config: LLMConfig,
    messages: list[dict],
    *,
    temperature: float | None = None,
    read_timeout: int | None = None,
) -> requests.Response:
    """Open a streamed ``/chat/completions`` request and return the live response.

    The connection is established and the HTTP status validated *before* the
    caller starts reading, so configuration/connection/HTTP errors surface here
    (as :class:`LLMError`) and can be turned into a proper error status instead
    of failing mid-stream. Iterate the returned response with :func:`iter_stream`.
    """
    if not config.is_ready:
        raise LLMError("Il modello LLM non è configurato. Aprilo dalle impostazioni ⚙️.")

    headers = {"Content-Type": "application/json"}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"

    payload = {
        "model": config.model,
        "messages": messages,
        "temperature": config.temperature if temperature is None else temperature,
        "stream": True,
    }
    timeout = (_CONNECT_TIMEOUT, read_timeout or _DEFAULT_READ_TIMEOUT)

    try:
        resp = requests.post(
            _endpoint(config.base_url),
            json=payload,
            headers=headers,
            stream=True,
            timeout=timeout,
        )
    except requests.RequestException as exc:
        raise LLMError(f"Impossibile contattare il modello: {exc}") from exc

    if resp.status_code >= 400:
        detail = _error_detail(resp)
        resp.close()
        raise LLMError(f"Il modello ha risposto con errore {resp.status_code}: {detail}")

    # OpenAI-compatible SSE payloads are UTF-8. If the server omits the charset,
    # requests would default to Latin-1 and mangle accented characters, so pin it.
    resp.encoding = "utf-8"
    return resp


def iter_stream(resp: requests.Response):
    """Yield text pieces from an OpenAI-compatible SSE stream of chat deltas.

    Swallows a mid-stream transport error (the caller keeps whatever was already
    yielded); always closes the response.
    """
    try:
        for raw in resp.iter_lines(decode_unicode=True):
            if not raw:
                continue
            line = raw.strip()
            if line.startswith("data:"):
                line = line[len("data:") :].strip()
            if line == "[DONE]":
                break
            if not line:
                continue
            try:
                chunk = json.loads(line)
            except ValueError:
                continue  # keep-alive or non-JSON line
            piece = _delta_text(chunk)
            if piece:
                yield piece
    except requests.RequestException:
        return  # a stall after partial output — stop, keep what was yielded
    finally:
        resp.close()


def chat(
    config: LLMConfig,
    messages: list[dict],
    *,
    temperature: float | None = None,
    read_timeout: int | None = None,
) -> str:
    """Non-streaming convenience wrapper: assemble the full answer as one string."""
    resp = open_stream(config, messages, temperature=temperature, read_timeout=read_timeout)
    text = "".join(iter_stream(resp)).strip()
    if not text:
        raise LLMError("Il modello non ha restituito alcun testo.")
    return text


def _delta_text(chunk: dict) -> str:
    """Extract the incremental text from one streamed chat-completion chunk."""
    try:
        choice = chunk["choices"][0]
    except (KeyError, IndexError, TypeError):
        return ""
    delta = choice.get("delta") or {}
    # Streaming uses ``delta.content``; some servers echo a final ``message``.
    content = delta.get("content")
    if content is None and isinstance(choice.get("message"), dict):
        content = choice["message"].get("content")
    return content or ""


def _error_detail(resp: requests.Response) -> str:
    try:
        body = resp.json()
        if isinstance(body, dict):
            err = body.get("error")
            if isinstance(err, dict):
                return str(err.get("message") or err)
            if err:
                return str(err)
        return str(body)[:300]
    except ValueError:
        return (resp.text or "").strip()[:300] or "nessun dettaglio"
