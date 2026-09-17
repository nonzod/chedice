"""Small shared helpers."""

from __future__ import annotations

import re
from pathlib import Path

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def safe_name(name: str) -> str:
    """Sanitize a string for safe use inside a filename."""
    return _UNSAFE.sub("_", name)


def safe_stem(name: str) -> str:
    """Derive a filesystem-safe base name (no extension) for downloads."""
    stem = Path(name).stem
    cleaned = _UNSAFE.sub("_", stem).strip("_")
    return cleaned or "transcript"
