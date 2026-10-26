"""Streamlit demo for Agent Memory Lab — Streamlit Community Cloud entrypoint."""

from __future__ import annotations

import base64
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import streamlit as st

from agent_memory_lab import MemoryLab
from agent_memory_lab.eval import run_eval, score_summary

AUTHOR = "Viraj Rana"
AUTHOR_TITLE = "ML Engineer"
DOCS = ROOT / "data" / "sample_docs"

st.set_page_config(
    page_title="Agent Memory Lab | RAG & AI Agent Memory by Viraj Rana",
    page_icon="🧠",
    layout="wide",
    menu_items={
        "About": (
            "Agent Memory Lab — RAG / agent-memory demo with citations. "
            "Built by Viraj Rana, ML Engineer. "
            "https://github.com/Viraj-rana/agent-memory-lab"
        ),
        "Get help": "https://github.com/Viraj-rana/agent-memory-lab#readme",
        "Report a bug": "https://github.com/Viraj-rana/agent-memory-lab/issues",
    },
)


@st.cache_resource(show_spinner="Indexing sample memory…")
def get_lab() -> MemoryLab:
    lab = MemoryLab()
    lab.ingest_folder(DOCS)
    return lab


def main() -> None:
    lab = get_lab()
    photo = ROOT / "docs" / "assets" / "viraj-rana-ml-engineer-agent-memory-lab.jpg"

    with st.sidebar:
        if photo.exists():
            # Small circular avatar (not full-width sidebar image)
            b64 = base64.b64encode(photo.read_bytes()).decode("ascii")
            st.markdown(
                f"""
                <div style="display:flex;align-items:center;gap:0.65rem;margin:0.25rem 0 0.75rem;">
                  <img
                    src="data:image/jpeg;base64,{b64}"
                    alt="{AUTHOR}, {AUTHOR_TITLE}"
                    width="52"
                    height="52"
                    style="width:52px;height:52px;border-radius:50%;object-fit:cover;border:2px solid #818cf8;"
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
        st.link_button(
            "Live demo home",
            "https://agent-memory-lab-nmyqs8dvibpgt28psyn2tl.streamlit.app/",
        )
        st.link_button("GitHub repo", "https://github.com/Viraj-rana/agent-memory-lab")

    st.title("Agent Memory Lab")
    st.caption(
        f"RAG & AI agent memory demo with citations · no API key required · "
        f"**{AUTHOR}** · {AUTHOR_TITLE}"
    )
    st.markdown(
        "Explore **retrieval-augmented generation**: ask questions, inspect "
        "**citations**, and run a golden **evaluation** suite."
    )
    st.write(
        f"Indexed **{lab.chunk_count}** chunks from "
        f"`{', '.join(lab.doc_ids)}`."
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
        top_k = st.slider("Top-K chunks", min_value=1, max_value=6, value=4)
        ask = st.button("Ask", type="primary", use_container_width=True)

        examples = [
            "What is agent memory and why does it matter?",
            "How does hybrid retrieval combine signals?",
            "What should a citation include in a RAG answer?",
            "How do you evaluate a memory system quickly?",
        ]
        picked = st.selectbox("Examples", options=["—"] + examples)
        if picked != "—":
            question = picked

        if ask or st.session_state.get("_auto_ask"):
            result = lab.ask(question, top_k=top_k, prefer_llm=False)
            st.markdown(f"**Mode:** `{result.mode}`")
            st.markdown(result.answer)
            st.subheader("Citations")
            if not result.citations:
                st.info("No citations")
            else:
                for i, c in enumerate(result.citations, start=1):
                    excerpt = c.text if len(c.text) < 280 else c.text[:277] + "..."
                    st.markdown(f"**[{i}]** `{c.doc_id}` · score `{c.score}`")
                    st.markdown(f"> {excerpt}")

    with col_eval:
        st.subheader("Golden eval")
        if st.button("Run golden eval", use_container_width=True):
            results = run_eval(lab)
            summary = score_summary(results)
            st.metric("Text hit rate", f"{summary['text_hit_rate']:.0%}")
            st.metric("Citation hit rate", f"{summary['citation_hit_rate']:.0%}")
            rows = [
                {
                    "Question": r.question,
                    "Text": "✅" if r.pass_text else "❌",
                    "Citation": "✅" if r.pass_citation else "❌",
                    "Docs": ", ".join(r.cited_docs) or "—",
                }
                for r in results
            ]
            st.dataframe(rows, use_container_width=True, hide_index=True)

    st.divider()
    st.markdown(
        f"Built by **{AUTHOR}** ({AUTHOR_TITLE}) · source in this repository · MIT"
    )


if __name__ == "__main__":
    main()
