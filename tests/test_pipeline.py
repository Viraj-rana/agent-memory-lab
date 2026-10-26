from __future__ import annotations

from pathlib import Path

from agent_memory_lab import MemoryLab
from agent_memory_lab.eval import run_eval, score_summary

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "data" / "sample_docs"


def test_ingest_and_ask_returns_citations():
    lab = MemoryLab()
    n = lab.ingest_folder(DOCS)
    assert n > 0
    ans = lab.ask("What is agent memory?", prefer_llm=False)
    assert ans.answer
    assert ans.citations
    assert ans.mode == "extractive"


def test_eval_hit_rates_are_strong():
    lab = MemoryLab()
    lab.ingest_folder(DOCS)
    results = run_eval(lab)
    summary = score_summary(results)
    assert summary["text_hit_rate"] >= 0.75
    assert summary["citation_hit_rate"] >= 0.75
