from __future__ import annotations

import os
import re
from typing import Callable

from .types import Answer, Citation, Chunk

_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def _extractive(question: str, hits: list[tuple[Chunk, float]]) -> Answer:
    """Build an answer from top sentences without an LLM."""
    if not hits:
        return Answer(
            question=question,
            answer="I could not find relevant memory for that question.",
            citations=(),
            mode="extractive",
        )

    citations = tuple(
        Citation(doc_id=c.doc_id, chunk_id=c.id, text=c.text, score=round(score, 4))
        for c, score in hits
    )
    # Prefer the densest sentences from the best hit, then stitch context.
    best_chunk, _ = hits[0]
    sentences = [s.strip() for s in _SENTENCE.split(best_chunk.text) if s.strip()]
    q_terms = {t.lower() for t in re.findall(r"[a-zA-Z0-9_]+", question) if len(t) > 2}

    def rank(sentence: str) -> int:
        tokens = {t.lower() for t in re.findall(r"[a-zA-Z0-9_]+", sentence)}
        return len(tokens & q_terms)

    ranked = sorted(sentences, key=rank, reverse=True)
    picked = ranked[:2] if ranked else [best_chunk.text]
    extras = [c.text for c, _ in hits[1:3]]
    answer = " ".join(picked)
    if extras:
        answer += "\n\nRelated context:\n- " + "\n- ".join(extras[:2])

    return Answer(question=question, answer=answer, citations=citations, mode="extractive")


def _llm_answer(question: str, hits: list[tuple[Chunk, float]]) -> Answer | None:
    """Optional OpenAI-compatible generation when OPENAI_API_KEY is set."""
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
    if not api_key or not hits:
        return None
    try:
        from urllib import request
        import json
    except ImportError:
        return None

    context = "\n\n".join(
        f"[{i + 1}] (doc={c.doc_id}) {c.text}" for i, (c, _) in enumerate(hits)
    )
    payload = {
        "model": os.getenv("LLM_MODEL", "gpt-4o-mini"),
        "messages": [
            {
                "role": "system",
                "content": (
                    "Answer using only the provided memory snippets. "
                    "Cite sources like [1], [2]. If unknown, say you do not know."
                ),
            },
            {
                "role": "user",
                "content": f"Question: {question}\n\nMemory:\n{context}",
            },
        ],
        "temperature": 0.2,
    }
    base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    req = request.Request(
        f"{base}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
        text = data["choices"][0]["message"]["content"].strip()
    except Exception:
        return None

    citations = tuple(
        Citation(doc_id=c.doc_id, chunk_id=c.id, text=c.text, score=round(score, 4))
        for c, score in hits
    )
    return Answer(question=question, answer=text, citations=citations, mode="llm")


def answer_question(
    question: str,
    hits: list[tuple[Chunk, float]],
    *,
    prefer_llm: bool = True,
) -> Answer:
    if prefer_llm:
        llm = _llm_answer(question, hits)
        if llm is not None:
            return llm
    return _extractive(question, hits)


AnswerFn = Callable[[str, list[tuple[Chunk, float]]], Answer]
