"""Streamlit demo for Agent Memory Lab — Streamlit Community Cloud entrypoint."""

from __future__ import annotations

import base64
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import streamlit as st

from agent_memory_lab import MemoryLab
from agent_memory_lab.eval import run_eval, score_summary
from agent_memory_lab.types import RetrievalMode

AUTHOR = "Viraj Rana"
AUTHOR_TITLE = "ML Engineer"
DOCS = ROOT / "data" / "sample_docs"
LIVE = "https://agent-memory-lab-nmyqs8dvibpgt28psyn2tl.streamlit.app/"

SECRET_KEYS = (
    "OPENAI_API_KEY",
    "LLM_API_KEY",
    "GROQ_API_KEY",
    "OPENAI_BASE_URL",
    "GROQ_BASE_URL",
    "LLM_MODEL",
    "GROQ_MODEL",
    "EMBEDDING_MODEL",
)

st.set_page_config(
    page_title="Agent Memory Lab | RAG & AI Agent Memory by Viraj Rana",
    page_icon="🧠",
    layout="wide",
    menu_items={
        "About": (
            "Agent Memory Lab — hybrid RAG / agent-memory demo with citations. "
            "Built by Viraj Rana, ML Engineer. "
            "https://github.com/Viraj-rana/agent-memory-lab"
        ),
        "Get help": "https://github.com/Viraj-rana/agent-memory-lab#readme",
        "Report a bug": "https://github.com/Viraj-rana/agent-memory-lab/issues",
    },
)


def apply_streamlit_secrets() -> None:
    """Copy Streamlit secrets into env so providers.py can read them safely."""
    try:
        secrets = st.secrets
    except Exception:
        return
    for key in SECRET_KEYS:
        try:
            value = secrets.get(key)
        except Exception:
            continue
        if value and key not in os.environ:
            os.environ[key] = str(value)


def get_lab(*, product_mode: bool) -> MemoryLab:
    apply_streamlit_secrets()
    if "lab" not in st.session_state:
        lab = MemoryLab(default_mode="hybrid", product_mode=product_mode)
        lab.ingest_folder(DOCS)
        st.session_state.lab = lab
    else:
        lab = st.session_state.lab
        lab.product_mode = product_mode
        lab.maybe_upgrade_dense()
    return lab


def main() -> None:
    apply_streamlit_secrets()
    photo = ROOT / "docs" / "assets" / "viraj-rana-ml-engineer-agent-memory-lab.jpg"

    with st.sidebar:
        if photo.exists():
            b64 = base64.b64encode(photo.read_bytes()).decode("ascii")
            st.markdown(
                f"""
                <div style="display:flex;align-items:center;gap:0.65rem;margin:0.25rem 0 0.75rem;">
                  <img
                    src="data:image/jpeg;base64,{b64}"
                    alt="{AUTHOR}, {AUTHOR_TITLE}"
                    width="88"
                    height="88"
                    style="width:88px;height:88px;border-radius:50%;object-fit:cover;border:2px solid #818cf8;"
                  />
                  <div>
                    <div style="font-weight:700;line-height:1.2;">{AUTHOR}</div>
                    <div style="opacity:0.7;font-size:0.85rem;">{AUTHOR_TITLE}</div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(f"**{AUTHOR}**")
            st.caption(AUTHOR_TITLE)

        st.link_button("Live demo home", LIVE)
        st.link_button("GitHub repo", "https://github.com/Viraj-rana/agent-memory-lab")
        st.divider()

        experience = st.radio(
            "Experience",
            options=["Product", "Lab"],
            index=0,
            help=(
                "Product uses LLM-RAG + neural embeddings when API keys are set. "
                "Lab always uses offline TF-IDF + LSA + extractive answers."
            ),
        )
        product_mode = experience == "Product"

        st.divider()
        st.subheader("Upload memory")
        uploads = st.file_uploader(
            "Add .md / .txt docs",
            type=["md", "txt", "markdown"],
            accept_multiple_files=True,
        )

        lab = get_lab(product_mode=product_mode)

        if uploads and st.button("Ingest uploads", use_container_width=True):
            added = 0
            for item in uploads:
                try:
                    added += lab.ingest_bytes(item.name, item.getvalue())
                except ValueError as exc:
                    st.error(str(exc))
            st.success(f"Ingested {added} new chunks.")
            st.rerun()

        if st.button("Reset to sample docs", use_container_width=True):
            lab.reset()
            lab.ingest_folder(DOCS)
            st.rerun()

        caps = lab.capabilities()
        st.divider()
        st.caption("Provider status")
        st.write(
            f"LLM: `{'ready' if caps['llm_ready'] else 'offline'}` "
            f"({caps['llm_provider']})  \n"
            f"Dense: `{caps['dense_backend']}` "
            f"({caps['embed_provider']})"
        )
        if product_mode and not caps["llm_ready"]:
            st.info(
                "Product mode is on, but no LLM key is set — using extractive fallback. "
                "Add `OPENAI_API_KEY` or `GROQ_API_KEY` in Streamlit secrets for LLM-RAG."
            )

    st.title("Agent Memory Lab")
    st.caption(
        f"Hybrid RAG & AI agent memory · citations · eval · observability · "
        f"**{AUTHOR}** · {AUTHOR_TITLE}"
    )
    st.markdown(
        "**Hybrid retrieval** (TF-IDF + neural embeddings or LSA), "
        "**RRF fusion**, **LLM-RAG or extractive** answers with citations, uploads, "
        "golden eval, and query logs. Offline-safe without API keys."
    )
    st.write(
        f"Indexed **{lab.chunk_count}** chunks from "
        f"`{', '.join(lab.doc_ids)}` · dense=`{lab.dense_backend}`."
    )

    col_ask, col_eval = st.columns([3, 2], gap="large")

    with col_ask:
        st.subheader("Ask the memory")
        question = st.text_area(
            "Question",
            value="What is agent memory and why does it matter?",
            height=90,
            label_visibility="collapsed",
            placeholder="e.g. Why do citations matter in RAG?",
        )
        c1, c2 = st.columns(2)
        with c1:
            top_k = st.slider("Top-K chunks", min_value=1, max_value=8, value=4)
        with c2:
            mode: RetrievalMode = st.selectbox(
                "Retrieval mode",
                options=["hybrid", "sparse", "dense"],
                index=0,
                help="hybrid = sparse + dense fused with RRF",
            )
        ask = st.button("Ask", type="primary", use_container_width=True)

        examples = [
            "What is agent memory and why does it matter?",
            "How does hybrid retrieval combine signals?",
            "What should a citation include in a RAG answer?",
            "How do you evaluate a memory system quickly?",
            "What stages make up a clean agent-memory architecture?",
        ]
        picked = st.selectbox("Examples", options=["—"] + examples)
        if picked != "—":
            question = picked

        if ask:
            result = lab.ask(
                question,
                top_k=top_k,
                prefer_llm=product_mode,
                mode=mode,
            )
            event = lab.logger.latest(1)
            latency = event[0].latency_ms if event else None
            st.markdown(
                f"**Answer:** `{result.mode}` · **retrieval:** `{mode}` · "
                f"**dense:** `{lab.dense_backend}`"
            )
            if latency is not None:
                st.caption(f"Latency: **{latency:.1f} ms**")
            st.markdown(result.answer)
            st.subheader("Citations")
            if not result.citations:
                st.info("No citations (abstained or weak retrieval).")
            else:
                for i, c in enumerate(result.citations, start=1):
                    excerpt = c.text if len(c.text) < 280 else c.text[:277] + "..."
                    st.markdown(f"**[{i}]** `{c.doc_id}` · score `{c.score}`")
                    st.markdown(f"> {excerpt}")

    with col_eval:
        st.subheader("Golden eval")
        st.caption("Eval always uses extractive answers for stable offline scoring.")
        if st.button("Run golden eval", use_container_width=True):
            results = run_eval(lab)
            summary = score_summary(results)
            m1, m2 = st.columns(2)
            m1.metric("Text hit rate", f"{summary['text_hit_rate']:.0%}")
            m2.metric("Citation hit rate", f"{summary['citation_hit_rate']:.0%}")
            m3, m4 = st.columns(2)
            m3.metric("Precision@k", f"{summary['precision_at_k']:.0%}")
            m4.metric("Abstain accuracy", f"{summary['abstain_accuracy']:.0%}")
            st.caption(
                f"Avg latency: {summary['avg_latency_ms']:.1f} ms · "
                f"cases: {int(summary['cases'])}"
            )
            rows = [
                {
                    "Question": r.question,
                    "Text": "✅" if r.pass_text else "❌",
                    "Citation": "✅" if r.pass_citation else "❌",
                    "P@k": f"{r.precision_at_k:.2f}",
                    "Abstain OK": "✅" if r.pass_abstain else "❌",
                    "ms": f"{r.latency_ms:.1f}",
                    "Docs": ", ".join(r.cited_docs) or "—",
                }
                for r in results
            ]
            st.dataframe(rows, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Observability")
    logs = lab.logger.as_rows(12)
    if logs:
        st.dataframe(logs, use_container_width=True, hide_index=True)
    else:
        st.caption("Ask a question to populate the query log (latency, mode, top docs).")

    with st.expander("Architecture (quick view)"):
        st.markdown(
            """
```text
Upload/Docs → Chunk → TF-IDF (sparse) + Embeddings/LSA (dense)
                         ↘         ↙
                      Reciprocal Rank Fusion
                              ↓
                 LLM-RAG (product) or Extractive (lab)
                              ↓
                    Citations + Eval + Query logger
```
            """
        )

    st.markdown(
        f"Built by **{AUTHOR}** ({AUTHOR_TITLE}) · hybrid RAG portfolio lab · MIT"
    )


if __name__ == "__main__":
    main()
