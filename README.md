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

# Agent Memory Lab — RAG & AI Agent Memory Demo

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/demo-Streamlit-FF4B4B.svg)](https://share.streamlit.io/deploy?repository=Viraj-rana/agent-memory-lab&branch=master&mainModule=streamlit_app.py)
[![GitHub pages](https://img.shields.io/badge/SEO-GitHub%20Pages-222.svg)](https://viraj-rana.github.io/agent-memory-lab/)

**Retrieval-augmented generation (RAG)** lab for **AI agent memory**: ingest documents, retrieve relevant chunks, answer with **citations**, and measure quality with a golden eval suite.

**Author:** [Viraj Rana](https://github.com/Viraj-rana) · ML Engineer  
**Repo:** https://github.com/Viraj-rana/agent-memory-lab  
**SEO landing page:** https://viraj-rana.github.io/agent-memory-lab/

> Default demo needs **no API key** (TF-IDF retrieval + extractive answers).

## Why this project

Searchable skills this portfolio shows:

- RAG / retrieval-augmented generation pipelines
- Agent memory architecture (ingest → index → retrieve → answer → evaluate)
- Citation-aware answers for trustworthy AI
- Lightweight evaluation (text hit rate + citation hit rate)
- Clean Python packaging + Streamlit product UI

## Live demo

### Permanent (Streamlit Community Cloud — free)

1. Open [Deploy on Streamlit](https://share.streamlit.io/deploy?repository=Viraj-rana/agent-memory-lab&branch=master&mainModule=streamlit_app.py)
2. Sign in with GitHub → **Deploy**
3. Share your `https://….streamlit.app` link with recruiters

### SEO site (GitHub Pages)

Enable **Settings → Pages → Deploy from branch → `/docs`**.  
Public URL: https://viraj-rana.github.io/agent-memory-lab/

## Quickstart

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Features

| Capability | Implementation |
|---|---|
| Document ingest | Markdown → overlapping chunks |
| Retrieval | TF-IDF cosine similarity (no cloud embeddings) |
| Answering | Extractive answers grounded in retrieved memory |
| Citations | Doc id, score, and excerpt on every answer |
| Evaluation | Golden questions with text + citation hit rates |

## Project structure

```text
streamlit_app.py       # Streamlit Cloud entrypoint (SEO page title/meta-friendly)
docs/                  # GitHub Pages SEO landing (meta, Open Graph, JSON-LD)
src/agent_memory_lab/  # Core library
data/sample_docs/      # Demo corpus about agent memory & RAG
notebooks/             # Jupyter walkthrough
tests/                 # Unit tests
```

## Topics / keywords

`rag` · `retrieval-augmented-generation` · `agent-memory` · `ai-agents` · `machine-learning` · `citations` · `information-retrieval` · `streamlit` · `ml-engineer-portfolio` · `viraj-rana`

## Tests

```bash
pip install pytest
pytest -q
```

## Author

**Viraj Rana** — ML Engineer  

Building practical AI systems: RAG, agent memory, evaluation, and clean demos recruiters can click.

## License

MIT
