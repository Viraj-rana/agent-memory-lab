from __future__ import annotations

from dataclasses import dataclass

from .pipeline import MemoryLab


@dataclass(frozen=True, slots=True)
class EvalCase:
    question: str
    must_include: tuple[str, ...]
    must_cite_doc: str | None = None
    should_abstain: bool = False


DEFAULT_CASES: tuple[EvalCase, ...] = (
    EvalCase(
        question="What is agent memory and why does it matter?",
        must_include=("memory",),
        must_cite_doc="01_agent_memory",
    ),
    EvalCase(
        question="How does hybrid retrieval combine signals?",
        must_include=("retriev",),
        must_cite_doc="02_retrieval",
    ),
    EvalCase(
        question="What should a citation include in a RAG answer?",
        must_include=("citation",),
        must_cite_doc="03_citations",
    ),
    EvalCase(
        question="How do you evaluate a memory system quickly?",
        must_include=("evaluat",),
        must_cite_doc="04_evaluation",
    ),
    EvalCase(
        question="What stages make up a clean agent-memory architecture?",
        must_include=("ingest",),
        must_cite_doc="05_architecture",
    ),
    EvalCase(
        question="Why does sparse retrieval help with rare tokens?",
        must_include=("sparse", "token"),
        must_cite_doc="02_retrieval",
    ),
    EvalCase(
        question="What is the capital of Atlantis and who is the emperor?",
        must_include=(),
        must_cite_doc=None,
        should_abstain=True,
    ),
)


@dataclass(frozen=True, slots=True)
class CaseResult:
    question: str
    pass_text: bool
    pass_citation: bool
    precision_at_k: float
    abstained: bool
    pass_abstain: bool
    cited_docs: tuple[str, ...]
    latency_ms: float


def _precision_at_k(cited: tuple[str, ...], expected: str | None, k: int = 4) -> float:
    if expected is None:
        return 1.0 if not cited else 0.0
    top = cited[:k]
    if not top:
        return 0.0
    hits = sum(1 for doc in top if doc == expected)
    return hits / len(top)


def run_eval(
    lab: MemoryLab,
    cases: tuple[EvalCase, ...] = DEFAULT_CASES,
    *,
    top_k: int = 4,
) -> list[CaseResult]:
    results: list[CaseResult] = []
    for case in cases:
        before = len(lab.logger.latest(1))
        ans = lab.ask(case.question, top_k=top_k, prefer_llm=False)
        events = lab.logger.latest(1)
        latency = events[0].latency_ms if events else 0.0
        # If logger was empty and still empty, latency stays 0 (shouldn't happen).
        _ = before

        blob = ans.answer.lower()
        abstained = (
            not ans.citations
            or "could not find relevant memory" in blob
            or "do not know" in blob
        )
        if case.should_abstain:
            pass_text = True
            pass_citation = True
            pass_abstain = abstained
            precision = 1.0 if abstained else 0.0
            cited: tuple[str, ...] = tuple(sorted({c.doc_id for c in ans.citations}))
        else:
            pass_text = (
                True
                if not case.must_include
                else all(token.lower() in blob for token in case.must_include)
            )
            cited = tuple(c.doc_id for c in ans.citations)
            pass_citation = case.must_cite_doc is None or case.must_cite_doc in cited
            pass_abstain = not abstained
            precision = _precision_at_k(cited, case.must_cite_doc, k=top_k)

        results.append(
            CaseResult(
                question=case.question,
                pass_text=pass_text,
                pass_citation=pass_citation,
                precision_at_k=precision,
                abstained=abstained,
                pass_abstain=pass_abstain,
                cited_docs=tuple(sorted(set(cited))),
                latency_ms=latency,
            )
        )
    return results


def score_summary(results: list[CaseResult]) -> dict[str, float]:
    if not results:
        return {
            "text_hit_rate": 0.0,
            "citation_hit_rate": 0.0,
            "precision_at_k": 0.0,
            "abstain_accuracy": 0.0,
            "avg_latency_ms": 0.0,
            "cases": 0.0,
        }
    return {
        "text_hit_rate": sum(r.pass_text for r in results) / len(results),
        "citation_hit_rate": sum(r.pass_citation for r in results) / len(results),
        "precision_at_k": sum(r.precision_at_k for r in results) / len(results),
        "abstain_accuracy": sum(r.pass_abstain for r in results) / len(results),
        "avg_latency_ms": sum(r.latency_ms for r in results) / len(results),
        "cases": float(len(results)),
    }
