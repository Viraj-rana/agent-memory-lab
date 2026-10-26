from __future__ import annotations

from dataclasses import dataclass

from .pipeline import MemoryLab


@dataclass(frozen=True, slots=True)
class EvalCase:
    question: str
    must_include: tuple[str, ...]
    must_cite_doc: str | None = None


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
)


@dataclass(frozen=True, slots=True)
class CaseResult:
    question: str
    pass_text: bool
    pass_citation: bool
    cited_docs: tuple[str, ...]


def run_eval(lab: MemoryLab, cases: tuple[EvalCase, ...] = DEFAULT_CASES) -> list[CaseResult]:
    results: list[CaseResult] = []
    for case in cases:
        ans = lab.ask(case.question, top_k=4)
        blob = ans.answer.lower()
        pass_text = all(token.lower() in blob for token in case.must_include)
        cited = tuple(sorted({c.doc_id for c in ans.citations}))
        pass_citation = case.must_cite_doc is None or case.must_cite_doc in cited
        results.append(
            CaseResult(
                question=case.question,
                pass_text=pass_text,
                pass_citation=pass_citation,
                cited_docs=cited,
            )
        )
    return results


def score_summary(results: list[CaseResult]) -> dict[str, float]:
    if not results:
        return {"text_hit_rate": 0.0, "citation_hit_rate": 0.0, "cases": 0.0}
    return {
        "text_hit_rate": sum(r.pass_text for r in results) / len(results),
        "citation_hit_rate": sum(r.pass_citation for r in results) / len(results),
        "cases": float(len(results)),
    }
