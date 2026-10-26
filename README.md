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

<p align="center">
  <img
    src="docs/assets/viraj-rana-ml-engineer-agent-memory-lab.jpg"
    alt="Viraj Rana, ML Engineer — creator of Agent Memory Lab RAG and AI agent memory project"
    width="180"
  />
</p>

<p align="center">
  <strong>Viraj Rana</strong> · ML Engineer<br/>
  Open-source <strong>RAG</strong> / <strong>AI agent memory</strong> lab with citations
</p>

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B.svg)](https://agent-memory-lab-nmyqs8dvibpgt28psyn2tl.streamlit.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![GitHub pages](https://img.shields.io/badge/SEO-GitHub%20Pages-222.svg)](https://viraj-rana.github.io/agent-memory-lab/)

**Retrieval-augmented generation (RAG)** lab for **AI agent memory**: ingest documents, retrieve relevant chunks, answer with **citations**, and measure quality with a golden eval suite.

**Author:** [Viraj Rana](https://github.com/Viraj-rana) · ML Engineer  
**Live demo:** https://agent-memory-lab-nmyqs8dvibpgt28psyn2tl.streamlit.app/  
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
streamlit_app.py       # Streamlit Cloud entrypoint
docs/                  # GitHub Pages SEO landing + author photo
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

<img
  src="docs/assets/viraj-rana-ml-engineer-agent-memory-lab.jpg"
  alt="Portrait of Viraj Rana, ML Engineer, Agent Memory Lab"
  width="120"
/>

**Viraj Rana** — ML Engineer  

Building practical AI systems: RAG, agent memory, evaluation, and clean demos recruiters can click.

## License

MIT
