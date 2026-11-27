"""Node'ları hata toleranslı hale getirir.

Bir node hata fırlatırsa akış durmaz: hata state'in `errors` alanına yazılır ve
o node'a özel güvenli bir varsayılan çıktı döndürülür.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from graph.report import basic_report
from graph.state import Critique, NodeError, Plan, SubQuestion

log = logging.getLogger(__name__)

Node = Callable[[Any], dict]
Fallback = Callable[[Any], dict]


def guarded(name: str, fn: Node, fallback: Fallback) -> Node:
    def wrapper(state: Any) -> dict:
        try:
            return fn(state)
        except Exception as e:  # noqa: BLE001 — bilerek geniş: hiçbir ajan akışı düşürmemeli
            log.exception("%s node'u hata verdi", name)
            sq = state.get("sub_question") if isinstance(state, dict) else None
            err = NodeError(
                node=name,
                message=f"{type(e).__name__}: {e}"[:500],
                round=state.get("round", 0) if isinstance(state, dict) else 0,
                sub_question_id=getattr(sq, "id", None),
            )
            out = fallback(state)
            out["errors"] = [*out.get("errors", []), err]
            return out

    wrapper.__name__ = getattr(fn, "__name__", name)
    return wrapper


# --------------------------------------------------------------------------- #
# Node'a özel güvenli varsayılanlar
# --------------------------------------------------------------------------- #


def planner_fallback(state: dict) -> dict:
    """Sorunun kendisini tek alt soru olarak kullanır; round yine de artar (döngü sınırı korunur)."""
    rnd = state.get("round", 0) + 1
    return {"round": rnd, "plan": Plan(sub_questions=[SubQuestion(id=f"r{rnd}q1", text=state["question"])])}


def searcher_fallback(task: dict) -> dict:
    return {"sources": []}


def reader_fallback(state: dict) -> dict:
    return {"facts": []}


def critic_fallback(state: dict) -> dict:
    """Karar verilemezse eldekiyle raporu yazmaya geç."""
    return {"critique": Critique(verdict="sufficient", notes="Critic hata verdi; mevcut bulgularla devam edildi.")}


def writer_fallback(state: dict) -> dict:
    return {"report": basic_report(
        state["question"], state.get("facts", []), state.get("sources", []),
        note="Rapor yazıcısı hata verdi; bulgular ham liste olarak gösteriliyor.",
    )}
