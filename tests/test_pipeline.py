from __future__ import annotations

from pathlib import Path

from agent_memory_lab import MemoryLab
from agent_memory_lab.eval import run_eval, score_summary
from agent_memory_lab.providers import provider_status
from agent_memory_lab.store import reciprocal_rank_fusion

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "data" / "sample_docs"


def test_ingest_and_ask_returns_citations():
    lab = MemoryLab(product_mode=True)
    n = lab.ingest_folder(DOCS)
    assert n > 0
    # Offline: product mode falls back to extractive without keys.
    ans = lab.ask("What is agent memory?", prefer_llm=False, mode="hybrid")
    assert ans.answer
    assert ans.citations
    assert ans.mode == "extractive"
    event = lab.logger.latest(1)[0]
    assert event.answer_mode == "extractive"
    assert event.dense_backend in {"lsa", "api", "none"}


def test_product_mode_falls_back_without_keys(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    lab = MemoryLab(product_mode=True)
    lab.ingest_folder(DOCS)
    ans = lab.ask("What is agent memory?")
    assert ans.mode == "extractive"
    assert lab.dense_backend == "lsa"
    caps = lab.capabilities()
    assert caps["llm_ready"] is False
    assert caps["embed_ready"] is False


def test_provider_status_offline(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    status = provider_status()
    assert status.llm_ready is False
    assert status.embed_ready is False
    assert status.embed_provider == "lsa-fallback"


def test_hybrid_and_sparse_both_retrieve():
    lab = MemoryLab()
    lab.ingest_folder(DOCS)
    hybrid = lab.retrieve("hybrid retrieval sparse dense", top_k=3, mode="hybrid")
    sparse = lab.retrieve("hybrid retrieval sparse dense", top_k=3, mode="sparse")
    assert hybrid
    assert sparse


def test_upload_bytes():
    lab = MemoryLab()
    lab.ingest_folder(DOCS)
    n = lab.ingest_bytes("note.md", b"# Hello\n\nAgent memory stores durable context.\n")
    assert n >= 1
    assert any(doc.startswith("upload_") for doc in lab.doc_ids)


def test_rrf_prefers_consensus():
    fused = reciprocal_rank_fusion([[0, 1, 2], [1, 0, 3]])
    assert fused[0][0] in {0, 1}


def test_eval_hit_rates_are_strong():
    lab = MemoryLab()
    lab.ingest_folder(DOCS)
    results = run_eval(lab)
    summary = score_summary(results)
    assert summary["text_hit_rate"] >= 0.7
    assert summary["citation_hit_rate"] >= 0.7
    assert summary["precision_at_k"] >= 0.3
    assert summary["abstain_accuracy"] >= 0.5
    assert summary["cases"] >= 7
