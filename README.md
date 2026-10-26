# Agent Memory Lab

Lightweight **agent memory / RAG** demo with citations and a tiny golden eval.

Recruiters: open the live demo, ask a question, click **Run golden eval**.

## Live demo

**Public preview (while the local server is running):**  
https://four-feet-vanish.loca.lt

> If the tunnel asks for a password/IP gate, open http://127.0.0.1:7861 locally or redeploy to Hugging Face Spaces (permanent).

### Run locally

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -e ".[dev]"
set GRADIO_SHARE=0
set PORT=7861
python app.py
```

### Permanent host (Hugging Face Spaces)

1. Create a new Space (Gradio / Docker).
2. Push this repo (or use the included `Dockerfile`).
3. Space URL becomes your always-on recruiter link.

## What it shows

| Capability | How |
|---|---|
| Ingest | Markdown docs → overlapping chunks |
| Retrieve | TF-IDF cosine search (no API key) |
| Answer | Extractive by default; optional LLM if `OPENAI_API_KEY` is set |
| Citations | Doc id + score + excerpt on every answer |
| Eval | Text hit rate + citation hit rate on 4 golden questions |

## Project layout

```text
app.py                 # Gradio UI (shareable live preview)
src/agent_memory_lab/  # clean library code
data/sample_docs/      # demo corpus
notebooks/             # thin walkthrough
tests/                 # regression tests
```

## Notebook

```bash
jupyter notebook notebooks/01_agent_memory_lab.ipynb
```

## Tests

```bash
pytest -q
```

## Design notes

- Business logic lives in `src/` — the notebook and Gradio app only orchestrate.
- Default path needs **no cloud key**, so the demo always boots.
- Optional LLM generation uses the OpenAI-compatible HTTP API when a key exists.

## License

MIT
