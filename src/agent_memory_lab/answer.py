from __future__ import annotations

import re
from typing import Callable

from .providers import chat_completion, llm_endpoint
from .types import Answer, Citation, Chunk

_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def llm_available() -> bool:
    return llm_endpoint() is not None


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


def _llm_rag_answer(question: str, hits: list[tuple[Chunk, float]]) -> Answer | None:
    """Product RAG: generate an answer grounded only in retrieved memory."""
    if not hits or not llm_available():
        return None

    context = "\n\n".join(
        f"[{i + 1}] (doc={c.doc_id}, score={score:.3f})\n{c.text}"
        for i, (c, score) in enumerate(hits)
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You are a careful RAG assistant for an agent-memory system. "
                "Use ONLY the provided memory snippets. "
                "Cite supporting snippets inline like [1] or [2]. "
                "If the memory is insufficient, say you do not know and do not invent facts. "
                "Be concise and technical."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question:\n{question}\n\n"
                f"Retrieved memory:\n{context}\n\n"
                "Write the best answer you can from this memory, with citations."
            ),
        },
    ]
    text = chat_completion(messages, temperature=0.2)
    if not text:
        return None

    citations = tuple(
        Citation(doc_id=c.doc_id, chunk_id=c.id, text=c.text, score=round(score, 4))
        for c, score in hits
    )
    return Answer(question=question, answer=text, citations=citations, mode="llm-rag")


def answer_question(
    question: str,
    hits: list[tuple[Chunk, float]],
    *,
    prefer_llm: bool = True,
) -> Answer:
    if prefer_llm:
        llm = _llm_rag_answer(question, hits)
        if llm is not None:
            return llm
    return _extractive(question, hits)


AnswerFn = Callable[[str, list[tuple[Chunk, float]]], Answer]
