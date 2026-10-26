from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from .types import Chunk, RetrievalMode


def reciprocal_rank_fusion(
    ranked_lists: list[list[int]],
    *,
    k: int = 60,
) -> list[tuple[int, float]]:
    """Merge rank lists with RRF: score = Σ 1 / (k + rank)."""
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, idx in enumerate(ranked, start=1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


@dataclass
class VectorStore:
    """Hybrid in-memory index: sparse TF-IDF + dense LSA, fused with RRF.

    No API key / no heavy embedding model — suitable for Streamlit Cloud.
    """

    chunks: list[Chunk] = field(default_factory=list)
    _vectorizer: TfidfVectorizer | None = field(default=None, init=False, repr=False)
    _sparse: np.ndarray | None = field(default=None, init=False, repr=False)
    _svd: TruncatedSVD | None = field(default=None, init=False, repr=False)
    _dense: np.ndarray | None = field(default=None, init=False, repr=False)

    def clear(self) -> None:
        self.chunks.clear()
        self._vectorizer = None
        self._sparse = None
        self._svd = None
        self._dense = None

    def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        self.chunks.extend(chunks)
        self._rebuild()

    def _rebuild(self) -> None:
        texts = [c.text for c in self.chunks]
        self._vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=12_000,
            sublinear_tf=True,
        )
        self._sparse = self._vectorizer.fit_transform(texts)

        n_samples, n_features = self._sparse.shape
        # LSA needs at least 2 samples; keep dense optional for tiny corpora.
        if n_samples < 2 or n_features < 2:
            self._svd = None
            self._dense = None
            return

        n_components = max(2, min(64, n_samples - 1, n_features - 1))
        self._svd = TruncatedSVD(n_components=n_components, random_state=42)
        dense = self._svd.fit_transform(self._sparse)
        self._dense = normalize(dense)

    def _top_indices(self, scores: np.ndarray, top_k: int) -> list[int]:
        if scores.size == 0:
            return []
        k = min(top_k, scores.size)
        if k >= scores.size:
            order = np.argsort(scores)[::-1]
        else:
            idx = np.argpartition(scores, -k)[-k:]
            order = idx[np.argsort(scores[idx])[::-1]]
        return [int(i) for i in order if float(scores[int(i)]) > 0]

    def search_sparse(self, query: str, *, top_k: int = 8) -> list[tuple[Chunk, float]]:
        if not self.chunks or self._vectorizer is None or self._sparse is None:
            return []
        q = self._vectorizer.transform([query])
        scores = cosine_similarity(q, self._sparse).ravel()
        return [(self.chunks[i], float(scores[i])) for i in self._top_indices(scores, top_k)]

    def search_dense(self, query: str, *, top_k: int = 8) -> list[tuple[Chunk, float]]:
        if (
            not self.chunks
            or self._vectorizer is None
            or self._svd is None
            or self._dense is None
        ):
            # Fall back to sparse if LSA is unavailable.
            return self.search_sparse(query, top_k=top_k)
        q_sparse = self._vectorizer.transform([query])
        q_dense = normalize(self._svd.transform(q_sparse))
        scores = (self._dense @ q_dense.T).ravel()
        return [(self.chunks[i], float(scores[i])) for i in self._top_indices(scores, top_k)]

    def search(
        self,
        query: str,
        *,
        top_k: int = 4,
        mode: RetrievalMode = "hybrid",
        candidate_k: int | None = None,
    ) -> list[tuple[Chunk, float]]:
        pool = candidate_k or max(top_k * 3, 12)
        if mode == "sparse":
            return self.search_sparse(query, top_k=top_k)[:top_k]
        if mode == "dense":
            return self.search_dense(query, top_k=top_k)[:top_k]

        sparse_hits = self.search_sparse(query, top_k=pool)
        dense_hits = self.search_dense(query, top_k=pool)
        if not sparse_hits and not dense_hits:
            return []

        id_to_chunk = {c.id: c for c, _ in sparse_hits + dense_hits}
        id_to_idx = {cid: i for i, cid in enumerate(id_to_chunk)}
        sparse_ranks = [id_to_idx[c.id] for c, _ in sparse_hits if c.id in id_to_idx]
        dense_ranks = [id_to_idx[c.id] for c, _ in dense_hits if c.id in id_to_idx]
        fused = reciprocal_rank_fusion([sparse_ranks, dense_ranks])

        # Blend RRF score with max channel score for readable UI values.
        channel_scores: dict[str, float] = {}
        for chunk, score in sparse_hits + dense_hits:
            channel_scores[chunk.id] = max(channel_scores.get(chunk.id, 0.0), score)

        chunks = list(id_to_chunk.values())
        results: list[tuple[Chunk, float]] = []
        for idx, rrf in fused[:top_k]:
            chunk = chunks[idx]
            results.append((chunk, round(float(rrf) + 0.15 * channel_scores.get(chunk.id, 0.0), 4)))
        return results
