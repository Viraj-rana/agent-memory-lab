"""Gradio live demo for Agent Memory Lab — Hugging Face Spaces ready."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import gradio as gr

from agent_memory_lab import MemoryLab
from agent_memory_lab.eval import run_eval, score_summary

DOCS = ROOT / "data" / "sample_docs"

lab = MemoryLab()
lab.ingest_folder(DOCS)

AUTHOR = "Viraj Rana"
AUTHOR_TITLE = "ML Engineer"


def format_answer(question: str, top_k: int) -> tuple[str, str]:
    result = lab.ask(question, top_k=int(top_k), prefer_llm=True)
    body = f"**Mode:** `{result.mode}`\n\n{result.answer}"
    if not result.citations:
        return body, "_No citations_"
    lines = []
    for i, c in enumerate(result.citations, start=1):
        excerpt = c.text if len(c.text) < 280 else c.text[:277] + "..."
        lines.append(
            f"**[{i}]** `{c.doc_id}` · score `{c.score}`\n\n> {excerpt}"
        )
    return body, "\n\n".join(lines)


def run_demo_eval() -> str:
    results = run_eval(lab)
    summary = score_summary(results)
    rows = [
        f"- text hit rate: **{summary['text_hit_rate']:.0%}**",
        f"- citation hit rate: **{summary['citation_hit_rate']:.0%}**",
        f"- cases: **{int(summary['cases'])}**",
        "",
        "| Question | Text | Citation | Docs |",
        "|---|---|---|---|",
    ]
    for r in results:
        q = r.question.replace("|", "/")
        rows.append(
            f"| {q} | {'✅' if r.pass_text else '❌'} | "
            f"{'✅' if r.pass_citation else '❌'} | {', '.join(r.cited_docs) or '—'} |"
        )
    return "\n".join(rows)


EXAMPLES = [
    ["What is agent memory and why does it matter?", 4],
    ["How does hybrid retrieval combine signals?", 4],
    ["What should a citation include in a RAG answer?", 4],
    ["How do you evaluate a memory system quickly?", 4],
]


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Agent Memory Lab") as demo:
        gr.Markdown(
            f"""
            # Agent Memory Lab
            Lightweight RAG with **citations** and a tiny **eval suite**.  
            No API key required (extractive answers).

            **Author:** {AUTHOR} · {AUTHOR_TITLE}
            """
        )
        gr.Markdown(
            f"Indexed **{lab.chunk_count} chunks** from `{', '.join(lab.doc_ids)}`."
        )
        with gr.Row():
            with gr.Column(scale=3):
                question = gr.Textbox(
                    label="Ask the memory",
                    lines=2,
                    placeholder="e.g. Why do citations matter in RAG?",
                )
                top_k = gr.Slider(1, 6, value=4, step=1, label="Top-K chunks")
                ask_btn = gr.Button("Ask", variant="primary")
            with gr.Column(scale=2):
                eval_btn = gr.Button("Run golden eval")
                eval_out = gr.Markdown(value="_Run eval to see hit rates._")

        answer = gr.Markdown(label="Answer")
        citations = gr.Markdown(label="Citations")
        gr.Examples(examples=EXAMPLES, inputs=[question, top_k])

        ask_btn.click(format_answer, inputs=[question, top_k], outputs=[answer, citations])
        question.submit(format_answer, inputs=[question, top_k], outputs=[answer, citations])
        eval_btn.click(run_demo_eval, outputs=[eval_out])
    return demo


demo = build_ui()
demo.queue()

if __name__ == "__main__":
    # Local / Docker: share off by default on Spaces-like hosts
    share = os.getenv("GRADIO_SHARE", "0") == "1"
    port = int(os.getenv("PORT", "7860"))
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=share,
        show_error=True,
    )
