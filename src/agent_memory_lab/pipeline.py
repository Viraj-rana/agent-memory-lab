from __future__ import annotations

from pathlib import Path

from .answer import answer_question
from .chunking import chunk_text, load_documents
from .store import VectorStore
from .types import Answer, Chunk


class MemoryLab:
    """End-to-end agent memory: ingest → retrieve → answer with citations."""

    def __init__(self, *, chunk_size: int = 420, chunk_overlap: int = 80) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.store = VectorStore()

    @property
    def chunk_count(self) -> int:
        return len(self.store.chunks)

    @property
    def doc_ids(self) -> list[str]:
        return sorted({c.doc_id for c in self.store.chunks})

    def reset(self) -> None:
        self.store.clear()

    def ingest_text(self, doc_id: str, text: str) -> list[Chunk]:
        chunks = chunk_text(
            doc_id,
            text,
            size=self.chunk_size,
            overlap=self.chunk_overlap,
        )
        self.store.add(chunks)
        return chunks

    def ingest_folder(self, folder: str | Path) -> int:
        path = Path(folder)
        docs = load_documents(path)
        total = 0
        for doc_id, text in docs.items():
            total += len(self.ingest_text(doc_id, text))
        return total

    def retrieve(self, query: str, *, top_k: int = 4) -> list[tuple[Chunk, float]]:
        return self.store.search(query, top_k=top_k)

    def ask(self, question: str, *, top_k: int = 4, prefer_llm: bool = True) -> Answer:
        hits = self.retrieve(question, top_k=top_k)
        return answer_question(question, hits, prefer_llm=prefer_llm)
