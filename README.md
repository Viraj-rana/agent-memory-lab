---
title: Agent Memory Lab
emoji: 🧠
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
license: mit
short_description: RAG agent memory demo with citations — by Viraj Rana (ML Engineer)
tags:
  - rag
  - agent-memory
  - retrieval
  - gradio
---

# Agent Memory Lab

Lightweight **agent memory / RAG** demo with citations and a tiny golden eval.

**Author:** Viraj Rana · ML Engineer

Recruiters: open the live Space, ask a question, click **Run golden eval**.

## What it shows

| Capability | How |
|---|---|
| Ingest | Markdown docs → overlapping chunks |
| Retrieve | TF-IDF cosine search (no API key) |
| Answer | Extractive by default; optional LLM if `OPENAI_API_KEY` is set |
| Citations | Doc id + score + excerpt on every answer |
| Eval | Text hit rate + citation hit rate on 4 golden questions |

## Run locally

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
set GRADIO_SHARE=0
python app.py
```

## Design notes

- Business logic lives in `src/` — the Gradio app only orchestrates.
- Default path needs **no cloud key**, so the demo always boots.
- Built by **Viraj Rana (ML Engineer)** as a portfolio RAG / agent-memory lab.

## License

MIT
