"""Searcher (Rol D). Tek bir alt soru için kaynak bulur; her alt soru için paralel çağrılır.

Girdi state değil SearchTask'tir: {"question", "sub_question"}.
TODO(D): tools/search.py'deki gerçek aramaları kullan (Tavily -> DDG yedeği, arXiv, Semantic Scholar).
"""

from config import settings
from graph.state import SearchTask, Source


def run(task: SearchTask) -> dict:
    sq = task["sub_question"]
    sources = [
        Source(
            url=f"https://example.com/{sq.id}/{n}",
            title=f"Örnek kaynak {n} — {sq.text[:40]}",
            snippet=f"'{sq.text}' hakkında örnek özet {n}.",
            kind="web",
            sub_question_id=sq.id,
        )
        for n in range(1, min(2, settings.sources_per_question) + 1)
    ]
    return {"sources": sources}
