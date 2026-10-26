# Design decisions — Agent Memory Lab

**Author:** Viraj Rana, ML Engineer

This note explains *why* the lab is shaped this way. Recruiters: this is the senior signal — tradeoffs, not just features.

## Goals

1. Demonstrate a real RAG / agent-memory loop end-to-end.
2. Stay **free to run** (Streamlit Community Cloud, no API key).
3. Prefer **measurable quality** (citations + eval) over chat theatrics.

## Architecture

```text
Docs / uploads
    → chunking (overlapping windows)
    → sparse index (TF-IDF)
    → dense index (TruncatedSVD / LSA on TF-IDF)
    → reciprocal rank fusion (RRF)
    → extractive answer + citations
    → golden eval + query logger
```

## Key tradeoffs

| Decision | Chose | Rejected (for now) | Why |
|---|---|---|---|
| Dense signal | LSA on TF-IDF | sentence-transformers / OpenAI embeddings | Cold start on free hosting; no API key; still a true hybrid story |
| Fusion | Reciprocal Rank Fusion | Weighted score sum only | Rank-based fusion is robust across score scales |
| Answering | Extractive by default | Always-on LLM generation | Demos must work without keys; LLMs remain optional |
| Uploads | `.md` / `.txt` | Full PDF/OCR stack | Keeps Cloud install light; enough for portfolio realism |
| State | Per-session lab in Streamlit | Shared global cache for uploads | Avoids cross-user corpus contamination on Cloud |

## Evaluation philosophy

A tiny golden set is more useful than an unread leaderboard:

- **Text hit** — required concepts appear in the answer
- **Citation hit** — expected document appears in citations
- **Precision@k** — how concentrated citations are on the right doc
- **Abstain accuracy** — refuse when memory cannot support the question
- **Latency** — logged per query for honesty about cost/speed

## What I would ship next in production

1. True dense embeddings + cross-encoder rerank
2. PDF/HTML loaders and ACL-aware document scopes
3. Persistent vector/graph store with migrations
4. Auth, rate limits, and structured tracing (OpenTelemetry)
5. Online evaluation from user feedback

## Non-goals (intentionally)

- Multi-tenant SaaS control plane
- Claiming “production agent OS”
- Dependency-heavy agent frameworks that obscure the retrieval loop
