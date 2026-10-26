# Citations in RAG

A citation links an answer claim back to the source chunk that supports it.
Without citations, users cannot audit the agent and trust collapses.

Each citation should include:
- document id
- chunk id or location
- relevance score
- a short excerpt

Citation quality is a product feature, not a formatting nicety. Prefer answers
that abstain when memory is weak over confident hallucinations. Show the top
supporting excerpts beside the generated text so reviewers can verify claims
in seconds.
