from __future__ import annotations

import hashlib
import re
from pathlib import Path

from .types import Chunk

_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    return _WS.sub(" ", text).strip()


def chunk_text(doc_id: str, text: str, *, size: int = 420, overlap: int = 80) -> list[Chunk]:
    """Split text into overlapping character windows with stable ids."""
    clean = normalize(text)
    if not clean:
        return []

    step = max(size - overlap, 1)
    chunks: list[Chunk] = []
    start = 0
    index = 0
    while start < len(clean):
        end = min(start + size, len(clean))
        # Prefer breaking on sentence boundaries when possible.
        window = clean[start:end]
        if end < len(clean):
            punct = max(window.rfind(". "), window.rfind("? "), window.rfind("! "))
            if punct >= size // 3:
                end = start + punct + 1
                window = clean[start:end]

        body = window.strip()
        if body:
            digest = hashlib.sha1(f"{doc_id}:{index}:{body}".encode()).hexdigest()[:10]
            chunks.append(
                Chunk(id=f"{doc_id}-{index}-{digest}", doc_id=doc_id, text=body, index=index)
            )
            index += 1
        if end >= len(clean):
            break
        start += max(end - start - overlap, step)

    return chunks


def load_documents(folder: Path) -> dict[str, str]:
    """Load .md / .txt files keyed by stem."""
    docs: dict[str, str] = {}
    for path in sorted(folder.glob("*")):
        if path.suffix.lower() not in {".md", ".txt"}:
            continue
        docs[path.stem] = path.read_text(encoding="utf-8")
    return docs
