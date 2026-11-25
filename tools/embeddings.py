"""Chunking ve embedding ile alaka sıralaması (Rol B)."""

from __future__ import annotations

import math

from llm.client import get_llm


def chunk(text: str, size: int = 1200, overlap: int = 200) -> list[str]:
    """Metni karakter bazlı, örtüşmeli parçalara böler."""
    if not text:
        return []
    step = max(1, size - overlap)
    return [text[i : i + size] for i in range(0, len(text), step)]


def _cos(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


def top_chunks(query: str, chunks: list[str], k: int = 3) -> list[str]:
    """Sorguya en alakalı k parçayı döndürür."""
    if len(chunks) <= k:
        return chunks
    vecs = get_llm().embed([query, *chunks])
    q, rest = vecs[0], vecs[1:]
    ranked = sorted(zip(chunks, rest), key=lambda p: _cos(q, p[1]), reverse=True)
    return [c for c, _ in ranked[:k]]
