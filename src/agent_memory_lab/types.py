from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Chunk:
    id: str
    doc_id: str
    text: str
    index: int


@dataclass(frozen=True, slots=True)
class Citation:
    doc_id: str
    chunk_id: str
    text: str
    score: float


@dataclass(frozen=True, slots=True)
class Answer:
    question: str
    answer: str
    citations: tuple[Citation, ...] = field(default_factory=tuple)
    mode: str = "extractive"
