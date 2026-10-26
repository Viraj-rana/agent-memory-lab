---
title: Agent Memory Lab
emoji: 🧠
colorFrom: indigo
colorTo: blue
sdk: streamlit
sdk_version: 1.32.0
app_file: streamlit_app.py
pinned: false
license: mit
short_description: RAG agent memory demo with citations — by Viraj Rana (ML Engineer)
tags:
  - rag
  - agent-memory
  - retrieval-augmented-generation
  - machine-learning
  - streamlit
  - citations
  - portfolio
---

<p align="center">
  <strong>Viraj Rana</strong> · ML Engineer<br/>
  Hybrid RAG / AI agent memory lab with <strong>Product</strong> (LLM-RAG + embeddings) and offline <strong>Lab</strong> modes
</p>

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B.svg)](https://agent-memory-lab-nmyqs8dvibpgt28psyn2tl.streamlit.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![GitHub pages](https://img.shields.io/badge/SEO-GitHub%20Pages-222.svg)](https://viraj-rana.github.io/agent-memory-lab/)

**Retrieval-augmented generation (RAG)** lab for **AI agent memory**: ingest documents (including uploads), retrieve with **hybrid sparse+dense fusion**, answer with **citations** (LLM-RAG or extractive), log latency, and measure quality with a golden eval suite.

**Author:** [Viraj Rana](https://github.com/Viraj-rana) · ML Engineer  
**Live demo:** https://agent-memory-lab-nmyqs8dvibpgt28psyn2tl.streamlit.app/  
**Repo:** https://github.com/Viraj-rana/agent-memory-lab  
**SEO landing page:** https://viraj-rana.github.io/agent-memory-lab/  
**Design notes:** [DESIGN.md](DESIGN.md)

> Default **Lab** path needs **no API key** (TF-IDF + LSA + extractive).  
> Optional **Product** path: neural embeddings + LLM-RAG when Streamlit secrets / env keys are set.

## Architecture

```text
Docs / uploads → Chunking
                 ├─ Sparse: TF-IDF
                 └─ Dense: OpenAI embeddings (if keyed) or LSA
                        ↘     ↙
                   Reciprocal Rank Fusion
                            ↓
         Product: LLM-RAG (OpenAI/Groq)  |  Lab: extractive
                            ↓
              Citations + golden eval + query observability
```

## Why this project

Searchable skills this portfolio shows:

- Hybrid RAG (sparse TF-IDF + dense embeddings/LSA, fused with RRF)
- Product vs Lab modes with safe offline fallbacks
- Agent memory architecture (ingest → index → retrieve → answer → evaluate)
- Citation-aware answers and abstain behavior
- Evaluation beyond vibes: hit rates, Precision@k, abstain accuracy, latency
- Uploadable corpus + Streamlit product UI
- Explicit design tradeoffs ([DESIGN.md](DESIGN.md))

## Live demo

**Permanent recruiter link (Streamlit):**  
https://agent-memory-lab-nmyqs8dvibpgt28psyn2tl.streamlit.app/

### SEO site (GitHub Pages)

Enable **Settings → Pages → Deploy from branch → `/docs`** if not already.  
Public URL: https://viraj-rana.github.io/agent-memory-lab/  
(Includes author photo + Open Graph image for search/social previews.)

## Quickstart

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

### Optional Product mode (keys)

Copy `.streamlit/secrets.toml.example` → `.streamlit/secrets.toml` (gitignored), or set the same keys in **Streamlit Cloud → App settings → Secrets**:

- `OPENAI_API_KEY` — chat + embeddings
- or `GROQ_API_KEY` — chat only (dense stays LSA unless OpenAI key is also set)

Never commit real keys.

## Features

| Capability | Implementation |
|---|---|
| Document ingest | Sample markdown + `.md`/`.txt` uploads (batched rebuild) |
| Hybrid retrieval | TF-IDF + neural embeddings or LSA + RRF fusion |
| Answering | Product LLM-RAG; Lab extractive; automatic fallback |
| Citations | Doc id, score, and excerpt on every answer |
| Evaluation | Text/citation hits, Precision@k, abstain accuracy, latency |
| Observability | Per-query retrieval mode, answer mode, dense backend, latency |

## Project structure

```text
streamlit_app.py       # Streamlit Cloud entrypoint
docs/                  # GitHub Pages SEO landing + author photo
src/agent_memory_lab/  # Core library (providers, store, answer, eval)
data/sample_docs/      # Demo corpus about agent memory & RAG
notebooks/             # Jupyter walkthrough
tests/                 # Unit tests (offline, no keys required)
```

## Topics / keywords

`rag` · `retrieval-augmented-generation` · `agent-memory` · `ai-agents` · `machine-learning` · `citations` · `information-retrieval` · `streamlit` · `ml-engineer-portfolio` · `viraj-rana`

## Tests

```bash
pip install pytest
pytest -q
```

## Author

<img
  src="docs/assets/viraj-rana-ml-engineer-agent-memory-lab.jpg"
  alt="Portrait of Viraj Rana, ML Engineer, Agent Memory Lab"
  width="120"
/>

**Viraj Rana** — ML Engineer  

Building practical AI systems: RAG, agent memory, evaluation, and clean demos recruiters can click.

## License

MIT
