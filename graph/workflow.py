"""LangGraph akışı: Planner -> (paralel) Searcher -> Reader -> Critic -> [Planner | Writer]."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from agents import critic, planner, reader, searcher, writer
from config import settings
from graph import resilience as r
from graph.state import ResearchState, SearchTask
from graph.trace import start_run, traced


def fan_out_search(state: ResearchState) -> list[Send] | str:
    """Plandaki her alt soru için ayrı bir Searcher çalıştırır.

    Plan boşsa Send listesi de boş olur ve akış rapor üretmeden biterdi;
    bu yüzden doğrudan Writer'a gidilir.
    """
    plan = state.get("plan")
    if not plan or not plan.sub_questions:
        return "writer"
    return [
        Send("searcher", SearchTask(question=state["question"], sub_question=sq))
        for sq in plan.sub_questions
    ]


def route_after_critic(state: ResearchState) -> str:
    critique = state.get("critique")
    if critique and critique.verdict == "needs_more" and state.get("round", 0) < settings.max_rounds:
        return "planner"
    return "writer"


def _node(name: str, fn, fallback):
    """Hata toleransı (guarded) + loglama (traced). Trace, hata kaydı dahil son çıktıyı görür."""
    return traced(name, r.guarded(name, fn, fallback))


def build_graph():
    g = StateGraph(ResearchState)
    g.add_node("planner", _node("planner", planner.run, r.planner_fallback))
    g.add_node("searcher", _node("searcher", searcher.run, r.searcher_fallback))
    g.add_node("reader", _node("reader", reader.run, r.reader_fallback))
    g.add_node("critic", _node("critic", critic.run, r.critic_fallback))
    g.add_node("writer", _node("writer", writer.run, r.writer_fallback))

    g.add_edge(START, "planner")
    g.add_conditional_edges("planner", fan_out_search, ["searcher", "writer"])
    g.add_edge("searcher", "reader")
    g.add_edge("reader", "critic")
    g.add_conditional_edges("critic", route_after_critic, ["planner", "writer"])
    g.add_edge("writer", END)
    return g.compile()


def run_config() -> dict:
    return {"max_concurrency": settings.max_concurrency, "recursion_limit": 50}


def run_research(question: str) -> ResearchState:
    """Tek seferde çalıştırır ve son state'i döndürür."""
    start_run()
    return build_graph().invoke({"question": question, "round": 0}, config=run_config())


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "Lokal LLM'lerin araştırma asistanlarında kullanımı"
    final = run_research(q)
    print(final["report"].markdown)
