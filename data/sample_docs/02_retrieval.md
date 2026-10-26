# Retrieval Strategies

Retrieval finds the memory snippets most useful for the current question.

Sparse retrieval (TF-IDF / BM25) is strong on exact keywords and rare tokens.
Dense retrieval (embeddings) is strong on paraphrases and semantic similarity.
Hybrid retrieval combines both signals so neither failure mode dominates.

A practical hybrid recipe:
1. Fetch top-N sparse hits and top-N dense hits
2. Merge with reciprocal rank fusion or weighted scores
3. Re-rank the shortlist before generation

Latency budgets matter. For interactive agents, keep retrieval under ~100ms
locally, then spend the remaining budget on generation.
