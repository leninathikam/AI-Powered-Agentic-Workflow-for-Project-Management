"""Resolve OpenAI / Vocareum API credentials and create clients."""

from __future__ import annotations

import os
import re

from openai import OpenAI

OPENAI_DEFAULT_BASE_URL = "https://api.openai.com/v1"
VOCAREUM_BASE_URL = "https://openai.vocareum.com/v1"


def resolve_openai_base_url(api_key: str | None = None) -> str:
    """
    Choose the API base URL.

    Priority:
    1. OPENAI_BASE_URL env var (explicit override)
    2. Vocareum URL when the key starts with ``voc-``
    3. Standard OpenAI URL for ``sk-...`` and other keys
    """
    explicit = (os.getenv("OPENAI_BASE_URL") or "").strip()
    if explicit:
        return explicit.rstrip("/")

    key = (api_key or os.getenv("OPENAI_API_KEY") or "").strip()
    if key.startswith("voc-"):
        return VOCAREUM_BASE_URL
    return OPENAI_DEFAULT_BASE_URL


def create_openai_client(api_key: str) -> OpenAI:
    """Create an OpenAI client pointed at the correct provider for this key."""
    return OpenAI(base_url=resolve_openai_base_url(api_key), api_key=api_key)


def format_openai_config_error(exc: BaseException) -> str:
    """Turn provider 400/auth failures into actionable setup guidance."""
    raw = str(exc)
    lowered = raw.lower()
    looks_like_bad_key = any(
        token in lowered
        for token in (
            "key was not found",
            "invalid_api_key",
            "incorrect api key",
            "invalid api key",
            "authentication",
            "unauthorized",
            "incorrect api key provided",
        )
    )
    if not looks_like_bad_key:
        return f"Workflow failed: {exc}"

    key = (os.getenv("OPENAI_API_KEY") or "").strip()
    base = resolve_openai_base_url(key or None)
    key_hint = "missing"
    if key.startswith("sk-"):
        key_hint = "sk-... (OpenAI)"
    elif key.startswith("voc-"):
        key_hint = "voc-... (Vocareum)"
    elif key:
        key_hint = f"{key[:4]}... (unrecognized prefix)"

    return (
        "Workflow failed: the API key was rejected by the configured endpoint.\n\n"
        f"Detected key type: {key_hint}\n"
        f"Using base URL: {base}\n\n"
        "Fix in Streamlit secrets (or `.env` locally):\n"
        "- Standard OpenAI keys (`sk-...`) → leave OPENAI_BASE_URL unset "
        f"(defaults to `{OPENAI_DEFAULT_BASE_URL}`), or set it explicitly.\n"
        "- Vocareum keys (`voc-...`) → leave OPENAI_BASE_URL unset "
        f"(defaults to `{VOCAREUM_BASE_URL}`), or set it explicitly.\n"
        "- Do not mix an OpenAI key with the Vocareum URL (or the reverse).\n\n"
        f"Provider detail: {re.sub(r'\\s+', ' ', raw).strip()}"
    )
