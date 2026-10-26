from __future__ import annotations

import time
from pathlib import Path

from .answer import answer_question, llm_available
from .chunking import chunk_text, load_documents
from .observability import QueryEvent, QueryLogger
from .providers import provider_status
from .store import VectorStore
from .types import Answer, Chunk, RetrievalMode


class MemoryLab:
    """End-to-end agent memory: ingest → hybrid retrieve → answer with citations."""

    def __init__(
        self,
        *,
        chunk_size: int = 420,
        chunk_overlap: int = 80,
        default_mode: RetrievalMode = "hybrid",
        product_mode: bool = True,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.default_mode = default_mode
        # Product mode prefers LLM-RAG when keys exist; always falls back safely.
        self.product_mode = product_mode
        self.store = VectorStore()
        self.logger = QueryLogger()

    @property
    def chunk_count(self) -> int:
        return len(self.store.chunks)

    @property
    def doc_ids(self) -> list[str]:
        return sorted({c.doc_id for c in self.store.chunks})

    @property
    def dense_backend(self) -> str:
        return self.store.dense_backend

    def capabilities(self) -> dict[str, str | bool]:
        status = provider_status()
        return {
            "product_mode": self.product_mode,
            "llm_ready": status.llm_ready,
            "embed_ready": status.embed_ready,
            "llm_provider": status.llm_provider,
            "embed_provider": status.embed_provider,
            "dense_backend": self.dense_backend,
            "llm_model": status.llm_model,
            "embed_model": status.embed_model,
        }

    def reset(self) -> None:
        self.store.clear()
        self.logger.clear()

    def ingest_text(self, doc_id: str, text: str) -> list[Chunk]:
        chunks = chunk_text(
            doc_id,
            text,
            size=self.chunk_size,
            overlap=self.chunk_overlap,
        )
        self.store.add(chunks)
        return chunks

    def ingest_many(self, docs: dict[str, str]) -> int:
        """Ingest many docs with a single index rebuild (important for API embeddings)."""
        all_chunks: list[Chunk] = []
        for doc_id, text in docs.items():
            all_chunks.extend(
                chunk_text(
                    doc_id,
                    text,
                    size=self.chunk_size,
                    overlap=self.chunk_overlap,
                )
            )
        self.store.add(all_chunks)
        return len(all_chunks)

    def ingest_folder(self, folder: str | Path) -> int:
        return self.ingest_many(load_documents(Path(folder)))

    def ingest_bytes(self, filename: str, raw: bytes) -> int:
        """Ingest an uploaded .md/.txt (UTF-8) file; returns chunk count."""
        name = Path(filename).name
        stem = Path(filename).stem.replace(" ", "_") or "upload"
        suffix = Path(filename).suffix.lower()
        if suffix not in {".md", ".txt", ".markdown"}:
            raise ValueError("Only .md / .txt uploads are supported in this lab build.")
        text = raw.decode("utf-8", errors="ignore")
        if not text.strip():
            raise ValueError(f"Uploaded file is empty: {name}")
        doc_id = f"upload_{stem}"
        return len(self.ingest_text(doc_id, text))

    def maybe_upgrade_dense(self) -> bool:
        """If neural embeddings are configured but index is still LSA, rebuild once."""
        caps = self.capabilities()
        if caps["embed_ready"] and self.dense_backend != "api" and self.chunk_count:
            self.store.refresh_index()
            return self.dense_backend == "api"
        return False

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 4,
        mode: RetrievalMode | None = None,
    ) -> list[tuple[Chunk, float]]:
        return self.store.search(query, top_k=top_k, mode=mode or self.default_mode)

    def ask(
        self,
        question: str,
        *,
        top_k: int = 4,
        prefer_llm: bool | None = None,
        mode: RetrievalMode | None = None,
        score_threshold: float = 0.02,
    ) -> Answer:
        started = time.perf_counter()
        used_mode = mode or self.default_mode
        use_llm = self.product_mode if prefer_llm is None else prefer_llm
        # If product mode is on but no key, still run extractive without failing.
        if use_llm and not llm_available():
            use_llm = False

        hits = self.retrieve(question, top_k=top_k, mode=used_mode)
        hits = [(c, s) for c, s in hits if s >= score_threshold]
        answer = answer_question(question, hits, prefer_llm=use_llm)
        latency_ms = (time.perf_counter() - started) * 1000.0
        abstained = (
            not answer.citations
            or "could not find relevant memory" in answer.answer.lower()
            or "do not know" in answer.answer.lower()
        )
        self.logger.record(
            QueryEvent(
                question=question,
                mode=used_mode,
                answer_mode=answer.mode,
                dense_backend=self.dense_backend,
                latency_ms=round(latency_ms, 2),
                top_doc_ids=tuple(c.doc_id for c, _ in hits[:top_k]),
                top_scores=tuple(round(s, 4) for _, s in hits[:top_k]),
                citation_count=len(answer.citations),
                abstained=abstained,
            )
        )
        return answer
