# Design decisions — Agent Memory Lab

**Author:** Viraj Rana, ML Engineer

This note explains *why* the lab is shaped this way. Recruiters: this is the senior signal — tradeoffs, not just features.

## Goals

1. Demonstrate a real RAG / agent-memory loop end-to-end.
2. Stay **free to run** on Streamlit Community Cloud without requiring API keys.
3. Offer a **Product** path (neural embeddings + LLM-RAG) when keys are present.
4. Prefer **measurable quality** (citations + eval) over chat theatrics.

## Architecture

```text
Docs / uploads
    → chunking (overlapping windows)
    → sparse index (TF-IDF)
    → dense index (OpenAI embeddings if keyed, else TruncatedSVD / LSA)
    → reciprocal rank fusion (RRF)
    → answer:
         Product → LLM-RAG (OpenAI / Groq) with extractive fallback
         Lab     → extractive only
    → golden eval + query logger
```

## Modes

| Mode | Dense | Answer | Needs keys? |
|---|---|---|---|
| **Lab** | LSA (or API if already indexed) | Extractive | No |
| **Product** | API embeddings when `OPENAI_API_KEY` set, else LSA | LLM-RAG when chat key set, else extractive | Optional |

Secrets are read from env / Streamlit secrets (see `.streamlit/secrets.toml.example`). Real keys are never committed.

## Key tradeoffs

| Decision | Chose | Rejected (for now) | Why |
|---|---|---|---|
| Dense default | LSA, upgrade to API embeddings when keyed | Always require sentence-transformers / paid embeddings | Cold start on free hosting; still a true hybrid story offline |
| Fusion | Reciprocal Rank Fusion | Weighted score sum only | Rank-based fusion is robust across score scales |
| Answering | Product LLM-RAG + Lab extractive | Always-on LLM generation | Demos must work without keys; LLMs remain optional |
| Ingest rebuild | Batch folder ingest (one rebuild) | Per-file rebuild only | Avoids N embedding API calls on sample corpus load |
| Uploads | `.md` / `.txt` | Full PDF/OCR stack | Keeps Cloud install light; enough for portfolio realism |
| State | Per-session lab in Streamlit | Shared global cache for uploads | Avoids cross-user corpus contamination on Cloud |
| Eval | Force extractive path | Score live LLM answers | Deterministic offline golden metrics |

## Evaluation philosophy

A tiny golden set is more useful than an unread leaderboard:

- **Text hit** — required concepts appear in the answer
- **Citation hit** — expected document appears in citations
- **Precision@k** — how concentrated citations are on the right doc
- **Abstain accuracy** — refuse when memory cannot support the question
- **Latency** — logged per query for honesty about cost/speed

## What I would ship next in production

1. Cross-encoder rerank on the fused candidate pool
2. PDF/HTML loaders and ACL-aware document scopes
3. Persistent vector/graph store with migrations
4. Auth, rate limits, and structured tracing (OpenTelemetry)
5. Online evaluation from user feedback

## Non-goals (intentionally)

- Multi-tenant SaaS control plane
- Claiming “production agent OS”
- Dependency-heavy agent frameworks that obscure the retrieval loop
