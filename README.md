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

## Repository

https://github.com/Viraj-rana/agent-memory-lab

## Live demo (Streamlit Community Cloud — permanent & free)

Repo is already on GitHub. Finish the one-time deploy:

1. Open [Streamlit Cloud deploy](https://share.streamlit.io/deploy?repository=Viraj-rana/agent-memory-lab&branch=master&mainModule=streamlit_app.py)
2. Sign in with GitHub → **Deploy**
3. Your permanent recruiter link will be:  
   `https://<app-name>.streamlit.app`

### Temporary preview (only while this PC is running)

https://launch-erp-bids-heath.trycloudflare.com  
Local: http://127.0.0.1:8501

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
