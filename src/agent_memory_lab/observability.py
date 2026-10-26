from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class QueryEvent:
    question: str
    mode: str
    latency_ms: float
    top_doc_ids: tuple[str, ...]
    top_scores: tuple[float, ...]
    citation_count: int
    abstained: bool


@dataclass
class QueryLogger:
    """Ring buffer of recent retrieval/answer events for the demo UI."""

    maxlen: int = 50
    _events: deque[QueryEvent] = field(default_factory=deque, init=False)

    def __post_init__(self) -> None:
        self._events = deque(maxlen=self.maxlen)

    def record(self, event: QueryEvent) -> None:
        self._events.appendleft(event)

    def clear(self) -> None:
        self._events.clear()

    def latest(self, n: int = 10) -> list[QueryEvent]:
        return list(self._events)[:n]

    def as_rows(self, n: int = 10) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for event in self.latest(n):
            row = asdict(event)
            row["top_doc_ids"] = ", ".join(event.top_doc_ids) or "—"
            row["top_scores"] = ", ".join(f"{s:.3f}" for s in event.top_scores) or "—"
            rows.append(row)
        return rows
