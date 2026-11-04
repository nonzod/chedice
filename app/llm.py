"""LLM client with multiple providers and priority-based fallback.

Two provider types are supported:

- ``ollama``  — native Ollama API (``/api/chat``, NDJSON streaming).
- ``openai``  — any OpenAI-compatible ``/chat/completions`` API (OpenAI,
  OpenRouter, llama.cpp server, vLLM, LM Studio, Ollama's ``/v1``…), with an
  optional bearer API key.

Each role (the text assistant and the vision model) has its own ordered list of
providers, configured from the settings UI and persisted in SQLite (see
:class:`app.jobs.JobStore`). Requests go to the provider with the highest
priority; when fallback is enabled and a provider fails (unreachable, HTTP
error, empty or incomplete answer), the next one in priority order is tried.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)

PROVIDER_TYPES = ("ollama", "openai")

# Connection timeout (fast fail if the host is unreachable — and fast fallback)
# and the default read timeout, applied *between streamed chunks*, not to the
# whole generation: only a genuine stall (e.g. a very slow cold start) trips it.
_CONNECT_TIMEOUT = 10  # seconds
_DEFAULT_READ_TIMEOUT = 300  # seconds between chunks (survives model cold-start)
_TEST_TIMEOUT = 15  # seconds, for the "test connection" button


class LLMError(RuntimeError):
    """Raised when no model is configured or every provider failed."""


class LLMIncompleteError(LLMError):
    """Raised when a streamed answer did not finish normally."""

    def __init__(self, message: str, partial_text: str) -> None:
        super().__init__(message)
        self.partial_text = partial_text


# ---- Configuration ------------------------------------------------------
def _truthy(value: object) -> bool:
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _parse_float(value: object, fallback: float) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return fallback


def guess_type(base_url: str) -> str:
    """Best guess of the provider type from a URL (Ollama listens on :11434)."""
    parsed = urlparse(base_url.strip())
    if parsed.port == 11434 or "ollama" in (parsed.hostname or ""):
        if not parsed.path.rstrip("/").endswith("/v1"):
            return "ollama"
    return "openai"


@dataclass
class Provider:
    """One model endpoint inside a role's fallback chain."""

    id: str
    name: str = ""
    type: str = "openai"
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    enabled: bool = True
    priority: int = 1  # 1 = tried first

    @property
    def is_ready(self) -> bool:
        return bool(self.enabled and self.base_url and self.model)

    @property
    def label(self) -> str:
        if self.name:
            return self.name
        kind = "Ollama" if self.type == "ollama" else "OpenAI-compatibile"
        return f"{kind} · {self.model}" if self.model else kind

    @classmethod
    def from_dict(cls, data: dict, *, fallback_priority: int = 1) -> Provider:
        ptype = str(data.get("type") or "").strip().lower()
        base_url = str(data.get("base_url") or "").strip()
        try:
            priority = int(data.get("priority") or fallback_priority)
        except (TypeError, ValueError):
            priority = fallback_priority
        return cls(
            id=str(data.get("id") or "").strip() or uuid.uuid4().hex[:8],
            name=str(data.get("name") or "").strip()[:60],
            type=ptype if ptype in PROVIDER_TYPES else guess_type(base_url),
            base_url=base_url,
            api_key=str(data.get("api_key") or "").strip(),
            model=str(data.get("model") or "").strip(),
            enabled=_truthy(data.get("enabled", True)),
            priority=max(1, priority),
        )

    def to_storage(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "base_url": self.base_url,
            "api_key": self.api_key,
            "model": self.model,
            "enabled": self.enabled,
            "priority": self.priority,
        }

    def to_public(self) -> dict:
        """Settings-form view: the API key itself is never sent back."""
        data = self.to_storage()
        data.pop("api_key")
        data["has_api_key"] = bool(self.api_key)
        data["label"] = self.label
        return data


@dataclass
class LLMConfig:
    """Settings of one role (``llm`` = text assistant, ``vision`` = scenes)."""

    enabled: bool = False
    fallback: bool = True  # try the next provider when one fails
    temperature: float = 0.3
    providers: list[Provider] = field(default_factory=list)

    @classmethod
    def from_store(cls, raw: dict[str, str], prefix: str = "llm") -> LLMConfig:
        providers: list[Provider] = []
        stored = raw.get(f"{prefix}_providers")
        if stored:
            try:
                items = json.loads(stored)
            except ValueError:
                items = []
            for index, item in enumerate(items if isinstance(items, list) else []):
                if isinstance(item, dict):
                    providers.append(Provider.from_dict(item, fallback_priority=index + 1))
        elif raw.get(f"{prefix}_base_url"):
            # Legacy single-endpoint configuration (before provider lists existed).
            providers.append(
                Provider(
                    id="legacy",
                    type="openai",
                    base_url=str(raw.get(f"{prefix}_base_url") or "").strip(),
                    api_key=str(raw.get(f"{prefix}_api_key") or "").strip(),
                    model=str(raw.get(f"{prefix}_model") or "").strip(),
                )
            )
        return cls(
            enabled=_truthy(raw.get(f"{prefix}_enabled", "false")),
            fallback=_truthy(raw.get(f"{prefix}_fallback", "true")),
            temperature=_parse_float(raw.get(f"{prefix}_temperature"), 0.3),
            providers=sorted(providers, key=lambda p: p.priority),
        )

    def to_store_values(self, prefix: str) -> dict[str, str]:
        return {
            f"{prefix}_enabled": "true" if self.enabled else "false",
            f"{prefix}_fallback": "true" if self.fallback else "false",
            f"{prefix}_temperature": str(self.temperature),
            f"{prefix}_providers": json.dumps(
                [p.to_storage() for p in self.providers], ensure_ascii=False
            ),
        }

    def to_public_dict(self) -> dict:
        chain = self.chain
        return {
            "enabled": self.enabled,
            "fallback": self.fallback,
            "temperature": self.temperature,
            "providers": [p.to_public() for p in self.providers],
            "ready": self.is_ready,
            "active": chain[0].label if chain else None,
        }

    @property
    def chain(self) -> list[Provider]:
        """Providers to try, in priority order (only the first if fallback is off)."""
        ready = [p for p in sorted(self.providers, key=lambda p: p.priority) if p.is_ready]
        return ready if self.fallback else ready[:1]

    @property
    def is_ready(self) -> bool:
        return bool(self.enabled and self.chain)

    def find(self, provider_id: str) -> Provider | None:
        return next((p for p in self.providers if p.id == provider_id), None)


# ---- Role-specific endpoint rules ---------------------------------------
def _host(base_url: str) -> str:
    return (urlparse(base_url.strip()).hostname or "").lower()


def is_valid_url(base_url: str) -> bool:
    parsed = urlparse(base_url.strip())
    return parsed.scheme in ("http", "https") and bool(parsed.hostname)


def is_text_base_url(base_url: str) -> bool:
    """The text assistant may not target the dedicated vision-model service."""
    return is_valid_url(base_url) and _host(base_url) != "llama"


def is_vision_base_url(base_url: str) -> bool:
    """Vision analysis may not target the dedicated text-model service."""
    return is_valid_url(base_url) and _host(base_url) != "text-llm"


# ---- Endpoints --------------------------------------------------------------
def _openai_base(base_url: str) -> str:
    """Normalize an OpenAI-compatible base URL to its ``…/v1``-style root.

    Tolerates a full endpoint (``…/chat/completions``), a bare host (assumes
    ``/v1``, as llama.cpp/vLLM/Ollama do) or an explicit API root (``…/api/v1``).
    """
    base = base_url.strip().rstrip("/")
    if base.endswith("/chat/completions"):
        return base[: -len("/chat/completions")]
    if urlparse(base).path in ("", "/"):
        return f"{base}/v1"
    return base


def _ollama_base(base_url: str) -> str:
    """Root of an Ollama server (strips ``/api/chat``, ``/api`` or ``/v1`` suffixes)."""
    base = base_url.strip().rstrip("/")
    for suffix in ("/api/chat", "/api", "/v1"):
        if base.endswith(suffix):
            return base[: -len(suffix)]
    return base


def _openai_headers(provider: Provider) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if provider.api_key:
        headers["Authorization"] = f"Bearer {provider.api_key}"
    return headers


def _to_ollama_messages(messages: list[dict]) -> list[dict]:
    """Convert OpenAI-style messages (incl. multimodal content) to Ollama's format."""
    converted = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            texts, images = [], []
            for part in content:
                if part.get("type") == "text":
                    texts.append(part.get("text", ""))
                elif part.get("type") == "image_url":
                    url = (part.get("image_url") or {}).get("url", "")
                    images.append(url.split(",", 1)[1] if url.startswith("data:") else url)
            item = {"role": message["role"], "content": "\n".join(texts)}
            if images:
                item["images"] = images
            converted.append(item)
        else:
            converted.append({"role": message["role"], "content": content or ""})
    return converted


# ---- Streaming ----------------------------------------------------------
@dataclass
class LLMStream:
    """An open streamed response plus the provider that is serving it."""

    response: requests.Response
    provider: Provider


def _open_provider(
    provider: Provider,
    messages: list[dict],
    *,
    temperature: float,
    max_tokens: int | None,
    read_timeout: int | None,
) -> LLMStream:
    """Open a streamed chat request on one provider, validating the HTTP status."""
    timeout = (_CONNECT_TIMEOUT, read_timeout or _DEFAULT_READ_TIMEOUT)
    if provider.type == "ollama":
        url = f"{_ollama_base(provider.base_url)}/api/chat"
        options: dict = {"temperature": temperature}
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        payload = {
            "model": provider.model,
            "messages": _to_ollama_messages(messages),
            "stream": True,
            "options": options,
        }
        headers = {"Content-Type": "application/json"}
    else:
        url = f"{_openai_base(provider.base_url)}/chat/completions"
        payload = {
            "model": provider.model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        headers = _openai_headers(provider)

    try:
        resp = requests.post(url, json=payload, headers=headers, stream=True, timeout=timeout)
    except requests.RequestException as exc:
        raise LLMError(f"impossibile contattare il server ({exc.__class__.__name__})") from exc

    if resp.status_code >= 400:
        detail = _error_detail(resp)
        resp.close()
        raise LLMError(f"errore HTTP {resp.status_code}: {detail}")

    # Both formats are UTF-8. If the server omits the charset, requests would
    # default to Latin-1 and mangle accented characters, so pin it.
    resp.encoding = "utf-8"
    return LLMStream(resp, provider)


def _all_failed(errors: list[str]) -> LLMError:
    if len(errors) == 1:
        return LLMError(f"Il modello non ha risposto — {errors[0]}")
    return LLMError("Nessun provider ha risposto: " + "; ".join(errors))


def open_stream(
    config: LLMConfig,
    messages: list[dict],
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    read_timeout: int | None = None,
) -> LLMStream:
    """Open a streamed request on the first provider that accepts it.

    Providers are tried in priority order (only the first one when fallback is
    disabled). Connection and HTTP errors are raised here, before the caller
    starts reading, so they can become a proper error status instead of a
    broken stream. Iterate the result with :func:`iter_stream`.
    """
    if not config.is_ready:
        raise LLMError("Il modello LLM non è configurato. Aprilo dalle impostazioni ⚙️.")
    temp = config.temperature if temperature is None else temperature
    errors: list[str] = []
    for provider in config.chain:
        try:
            return _open_provider(
                provider, messages, temperature=temp, max_tokens=max_tokens,
                read_timeout=read_timeout,
            )
        except LLMError as exc:
            logger.warning("LLM provider %s failed: %s", provider.label, exc)
            errors.append(f"{provider.label}: {exc}")
    raise _all_failed(errors)


def iter_stream(
    stream: LLMStream,
    on_finish: Callable[[str | None], None] | None = None,
) -> Iterator[str]:
    """Yield text pieces from an open stream (OpenAI SSE or Ollama NDJSON).

    ``on_finish`` receives the finish reason (``"stop"``, ``"length"``…).
    Swallows a mid-stream transport error (the caller keeps whatever was already
    yielded); always closes the response.
    """
    parse = _ollama_chunk if stream.provider.type == "ollama" else _openai_chunk
    resp = stream.response
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
            piece, finish_reason = parse(chunk)
            if piece:
                yield piece
            if finish_reason is not None and on_finish is not None:
                on_finish(finish_reason)
    except requests.RequestException:
        return  # a stall after partial output — stop, keep what was yielded
    finally:
        resp.close()


def _openai_chunk(chunk: dict) -> tuple[str, str | None]:
    try:
        choice = chunk["choices"][0]
    except (KeyError, IndexError, TypeError):
        return "", None
    delta = choice.get("delta") or {}
    # Streaming uses ``delta.content``; some servers echo a final ``message``.
    content = delta.get("content")
    if content is None and isinstance(choice.get("message"), dict):
        content = choice["message"].get("content")
    reason = choice.get("finish_reason")
    return content or "", (str(reason) if reason is not None else None)


def _ollama_chunk(chunk: dict) -> tuple[str, str | None]:
    if chunk.get("error"):
        raise requests.RequestException(str(chunk["error"]))
    content = (chunk.get("message") or {}).get("content") or ""
    reason = None
    if chunk.get("done"):
        reason = str(chunk.get("done_reason") or "stop")
    return content, reason


def chat(
    config: LLMConfig,
    messages: list[dict],
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    read_timeout: int | None = None,
    require_complete: bool = False,
) -> str:
    """Return the full answer as one string, falling back across providers.

    Unlike :func:`open_stream`, the whole answer is validated here, so an empty
    or (with ``require_complete``) truncated answer also moves on to the next
    provider. If every provider fails, the best partial answer is kept in the
    raised :class:`LLMIncompleteError`.
    """
    if not config.is_ready:
        raise LLMError("Il modello LLM non è configurato. Aprilo dalle impostazioni ⚙️.")
    temp = config.temperature if temperature is None else temperature
    errors: list[str] = []
    best_partial: LLMIncompleteError | None = None

    for provider in config.chain:
        try:
            text = _chat_one(provider, messages, temp, max_tokens, read_timeout,
                             require_complete)
            if errors:
                logger.info("LLM fallback: answered by %s", provider.label)
            return text
        except LLMIncompleteError as exc:
            errors.append(f"{provider.label}: {exc}")
            if best_partial is None or len(exc.partial_text) > len(best_partial.partial_text):
                best_partial = exc
        except LLMError as exc:
            errors.append(f"{provider.label}: {exc}")
        logger.warning("LLM provider %s failed: %s", provider.label, errors[-1])

    if best_partial is not None:
        raise LLMIncompleteError(str(best_partial), best_partial.partial_text)
    raise _all_failed(errors)


def _chat_one(
    provider: Provider,
    messages: list[dict],
    temperature: float,
    max_tokens: int | None,
    read_timeout: int | None,
    require_complete: bool,
) -> str:
    stream = _open_provider(
        provider, messages, temperature=temperature, max_tokens=max_tokens,
        read_timeout=read_timeout,
    )
    finish_reason: str | None = None

    def record_finish(reason: str | None) -> None:
        nonlocal finish_reason
        finish_reason = reason

    text = "".join(iter_stream(stream, record_finish)).strip()
    if not text:
        raise LLMError("il modello non ha restituito alcun testo")
    if require_complete and finish_reason != "stop":
        if finish_reason == "length":
            detail = ("il modello ha raggiunto il limite massimo di token "
                      "prima di completare tutte le scene")
        elif finish_reason is None:
            detail = "la risposta si è interrotta senza segnale di completamento"
        else:
            detail = f"risposta terminata in modo incompleto ({finish_reason})"
        raise LLMIncompleteError(detail, text)
    return text


# ---- Connection test ----------------------------------------------------
def test_provider(provider: Provider) -> dict:
    """Check that a provider is reachable and lists the configured model.

    Returns ``{"ok": bool, "message": str, "models": [...]}``. Uses the model
    listing endpoint (``/api/tags`` for Ollama, ``/models`` for OpenAI-compatible
    APIs), which is cheap and needs no generation.
    """
    if not is_valid_url(provider.base_url):
        return {"ok": False, "message": "URL non valido.", "models": []}
    if provider.type == "ollama":
        url = f"{_ollama_base(provider.base_url)}/api/tags"
        headers = {}
    else:
        url = f"{_openai_base(provider.base_url)}/models"
        headers = _openai_headers(provider)
    try:
        resp = requests.get(url, headers=headers, timeout=(_CONNECT_TIMEOUT, _TEST_TIMEOUT))
    except requests.RequestException as exc:
        return {"ok": False, "message": f"Server non raggiungibile ({exc.__class__.__name__}).",
                "models": []}
    if resp.status_code in (401, 403):
        return {"ok": False, "message": "Accesso negato: controlla la API key.", "models": []}
    if resp.status_code >= 400:
        return {"ok": False, "message": f"Errore HTTP {resp.status_code}: {_error_detail(resp)}",
                "models": []}
    try:
        body = resp.json()
    except ValueError:
        return {"ok": False, "message": "Risposta non valida dal server.", "models": []}

    if provider.type == "ollama":
        models = [m.get("name") or m.get("model") for m in body.get("models", [])]
    else:
        models = [m.get("id") for m in body.get("data", []) if isinstance(m, dict)]
    models = sorted({m for m in models if m})

    if not provider.model:
        return {"ok": True, "message": f"Connesso: {len(models)} modelli disponibili.",
                "models": models}
    available = provider.model in models or (
        provider.type == "ollama" and f"{provider.model}:latest" in models
    )
    if models and not available:
        return {"ok": False,
                "message": f"Connesso, ma il modello «{provider.model}» non è disponibile.",
                "models": models}
    return {"ok": True, "message": "Connesso, modello disponibile.", "models": models}


def _error_detail(resp: requests.Response) -> str:
    try:
        body = resp.json()
        if isinstance(body, dict):
            err = body.get("error")
            if isinstance(err, dict):
                return str(err.get("message") or err)[:300]
            if err:
                return str(err)[:300]
        return str(body)[:300]
    except ValueError:
        return (resp.text or "").strip()[:300] or "nessun dettaglio"
