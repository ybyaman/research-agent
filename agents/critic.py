"""Critic (Rol A). Bulguların planı karşılayıp karşılamadığına karar verir.

Mock davranış: 1. turda bir eksik bildirir (geri döngüyü test etmek için), sonra yeterli der.
TODO(A): llm.structured(..., schema=Critique) ile kapsam / çelişki / kaynak kalitesi kontrolü.
"""

from graph.state import Critique, ResearchState


def run(state: ResearchState) -> dict:
    if state.get("round", 0) == 1:
        return {"critique": Critique(
            verdict="needs_more",
            gaps=[f"{state['question']} — farklı yaklaşımların karşılaştırması"],
            notes="Mock: karşılaştırma eksik.",
        )}
    return {"critique": Critique(verdict="sufficient", notes="Mock: yeterli.")}
