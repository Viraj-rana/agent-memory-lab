"""LLM + embedding providers (OpenAI-compatible). Offline-safe when unset."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from urllib import error, request

import numpy as np


@dataclass(frozen=True, slots=True)
class ProviderStatus:
    llm_ready: bool
    embed_ready: bool
    llm_provider: str
    embed_provider: str
    llm_model: str
    embed_model: str


def _openai_key() -> str | None:
    return os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY") or None


def _groq_key() -> str | None:
    return os.getenv("GROQ_API_KEY") or None


def llm_endpoint() -> tuple[str, str, str] | None:
    """Return (api_key, base_url, model) for chat completions, if configured."""
    groq = _groq_key()
    if groq:
        return (
            groq,
            os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/"),
            os.getenv("GROQ_MODEL", os.getenv("LLM_MODEL", "llama-3.1-8b-instant")),
        )
    openai = _openai_key()
    if openai:
        return (
            openai,
            os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
            os.getenv("LLM_MODEL", "gpt-4o-mini"),
        )
    return None


def embed_endpoint() -> tuple[str, str, str] | None:
    """Return (api_key, base_url, model) for embeddings, if configured.

    Groq is chat-only here; embeddings use OpenAI-compatible keys.
    """
    openai = _openai_key()
    if not openai:
        return None
    return (
        openai,
        os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
        os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
    )


def provider_status() -> ProviderStatus:
    llm = llm_endpoint()
    emb = embed_endpoint()
    return ProviderStatus(
        llm_ready=llm is not None,
        embed_ready=emb is not None,
        llm_provider=("groq" if _groq_key() else "openai") if llm else "none",
        embed_provider="openai" if emb else "lsa-fallback",
        llm_model=llm[2] if llm else "—",
        embed_model=emb[2] if emb else "lsa",
    )


def _post_json(url: str, api_key: str, payload: dict, timeout: float = 45.0) -> dict:
    req = request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def chat_completion(messages: list[dict[str, str]], *, temperature: float = 0.2) -> str | None:
    cfg = llm_endpoint()
    if cfg is None:
        return None
    api_key, base, model = cfg
    try:
        data = _post_json(
            f"{base}/chat/completions",
            api_key,
            {"model": model, "messages": messages, "temperature": temperature},
        )
        return data["choices"][0]["message"]["content"].strip()
    except (error.URLError, error.HTTPError, KeyError, TimeoutError, json.JSONDecodeError):
        return None


def embed_texts(texts: list[str]) -> np.ndarray | None:
    """Embed texts via OpenAI-compatible API. Returns shape (n, d) or None."""
    cfg = embed_endpoint()
    if cfg is None or not texts:
        return None
    api_key, base, model = cfg
    vectors: list[list[float]] = []
    # Keep batches small for free-tier reliability.
    batch_size = 64
    try:
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            data = _post_json(
                f"{base}/embeddings",
                api_key,
                {"model": model, "input": batch},
            )
            # API may return out-of-order; sort by index.
            ordered = sorted(data["data"], key=lambda row: row["index"])
            vectors.extend(row["embedding"] for row in ordered)
        return np.asarray(vectors, dtype=np.float32)
    except (error.URLError, error.HTTPError, KeyError, TimeoutError, json.JSONDecodeError, ValueError):
        return None
