"""Araştırmayı çalıştırmanın tek yolu. CLI, eval ve UI bunu kullanır.

    for event in stream_research("soru"):      # UI: adım adım
        print(event.message)

    state = run_research("soru")               # eval/CLI: yalnızca sonuç
"""

from __future__ import annotations

import time
from typing import Iterator

from config import settings
from graph.events import SUMMARIZERS, RunEvent, error_message
from graph.state import NodeError, ResearchState
from graph.trace import start_run
from graph.workflow import build_graph, run_config
from llm.client import get_llm

_STAT_KEYS = ("calls", "failed_calls", "prompt_tokens", "output_tokens", "seconds", "wait_seconds")


def stream_research(question: str) -> Iterator[RunEvent]:
    """Her node bittiğinde bir RunEvent üretir; son olay `run_finished` ve raporu içerir."""
    run_id = start_run()
    t0 = time.perf_counter()
    seq = 0
    rnd = 0
    ctx: dict = {"sub_questions": {}}
    llm_before = get_llm().stats_snapshot()
    final: ResearchState = {}
    errors: list[NodeError] = []

    def emit(kind, message, node=None, data=None, final_state=None) -> RunEvent:
        nonlocal seq
        ev = RunEvent(run_id=run_id, seq=seq, kind=kind, node=node, round=rnd,
                      elapsed=round(time.perf_counter() - t0, 3), message=message, data=data or {},
                      final_state=final_state)
        seq += 1
        return ev

    yield emit("run_started", f"Araştırma başladı: {question}",
               data={"question": question, "mode": "mock" if settings.is_mock else "real",
                     "model": settings.llm_model})

    graph = build_graph()
    for mode, chunk in graph.stream({"question": question, "round": 0}, config=run_config(),
                                    stream_mode=["updates", "values"]):
        if mode == "values":
            final = chunk
            continue
        for node, out in chunk.items():
            out = out or {}
            if node == "planner" and "round" in out:
                rnd = out["round"]
            message, data = SUMMARIZERS[node](out, ctx)
            yield emit("node_finished", message, node=node, data=data)
            for err in out.get("errors", []):
                errors.append(err)
                yield emit("node_error", error_message(err), node=node, data=err.model_dump())

    llm_after = get_llm().stats_snapshot()
    report = final.get("report")
    yield emit(
        "run_finished",
        "Araştırma tamamlandı" if report else "Araştırma rapor üretmeden bitti",
        data={
            "report": report.model_dump() if report else None,
            "rounds": final.get("round", 0),
            "source_count": len(final.get("sources", [])),
            "fact_count": len(final.get("facts", [])),
            "error_count": len(errors),
            "llm": {k: round(llm_after[k] - llm_before[k], 3) for k in _STAT_KEYS},
        },
        final_state=final,
    )


def run_research(question: str) -> ResearchState:
    """Akışı sonuna kadar çalıştırır ve son state'i döndürür."""
    for event in stream_research(question):
        if event.kind == "run_finished":
            return event.final_state
    raise RuntimeError("Akış run_finished olayı üretmeden bitti")


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "Lokal LLM'lerin araştırma asistanlarında kullanımı"
    for ev in stream_research(q):
        print(f"[{ev.elapsed:6.1f}s] tur {ev.round} {ev.kind:<13} {ev.node or '':<8} {ev.message}")
        if ev.kind == "run_finished" and ev.data["report"]:
            print("\n" + ev.data["report"]["markdown"])
