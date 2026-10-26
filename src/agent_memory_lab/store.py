from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .types import Chunk


@dataclass
class VectorStore:
    """In-memory TF-IDF index — no API key, fast cold start for demos."""

    chunks: list[Chunk] = field(default_factory=list)
    _vectorizer: TfidfVectorizer | None = field(default=None, init=False, repr=False)
    _matrix: np.ndarray | None = field(default=None, init=False, repr=False)

    def clear(self) -> None:
        self.chunks.clear()
        self._vectorizer = None
        self._matrix = None

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
        self._matrix = self._vectorizer.fit_transform(texts)

    def search(self, query: str, *, top_k: int = 4) -> list[tuple[Chunk, float]]:
        if not self.chunks or self._vectorizer is None or self._matrix is None:
            return []
        q = self._vectorizer.transform([query])
        scores = cosine_similarity(q, self._matrix).ravel()
        if top_k >= len(scores):
            order = np.argsort(scores)[::-1]
        else:
            # Partial sort is enough for small demos and scales better.
            idx = np.argpartition(scores, -top_k)[-top_k:]
            order = idx[np.argsort(scores[idx])[::-1]]
        results: list[tuple[Chunk, float]] = []
        for i in order:
            score = float(scores[i])
            if score <= 0:
                continue
            results.append((self.chunks[int(i)], score))
        return results
