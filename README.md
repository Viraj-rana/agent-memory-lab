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
  - retrieval
  - streamlit
---

# Agent Memory Lab

Lightweight **agent memory / RAG** demo with citations and a tiny golden eval.

**Author:** Viraj Rana · ML Engineer

> No API keys required for the default extractive demo.

## Live demo (Streamlit Community Cloud)

1. Push this repo to GitHub (public).
2. Open [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Select `Viraj-rana/agent-memory-lab`, branch `master`, file `streamlit_app.py`.
4. Deploy — your permanent link looks like:  
   `https://<app-name>.streamlit.app`

## Run locally

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## What it shows

| Capability | How |
|---|---|
| Ingest | Markdown docs → overlapping chunks |
| Retrieve | TF-IDF cosine search (no API key) |
| Answer | Extractive answers with citations |
| Eval | Text + citation hit rates on golden questions |

## Project layout

```text
streamlit_app.py       # Streamlit Cloud entrypoint
app.py                 # Optional Gradio UI (local)
src/agent_memory_lab/  # library code
data/sample_docs/      # demo corpus
notebooks/             # walkthrough
tests/                 # unit tests
```

## Tests

```bash
pip install pytest
pytest -q
```

## Author

**Viraj Rana** — ML Engineer

## License

MIT
