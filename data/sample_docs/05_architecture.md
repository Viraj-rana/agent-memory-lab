# Mini Architecture

A clean agent-memory lab separates concerns:

- Ingest: normalize documents and create overlapping chunks
- Index: build a searchable vector / lexical store
- Retrieve: return ranked chunks for a query
- Answer: compose a response with citations
- Evaluate: measure text and citation hit rates

Keeping these stages as small modules makes demos honest and production ports
easier. The notebook should only orchestrate; business logic lives in packages.
This Agent Memory Lab follows that shape so recruiters can read the code path
in under five minutes.
