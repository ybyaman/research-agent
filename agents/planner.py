"""Planner (Rol C). Soruyu alt sorulara böler; 2. turda Critic'in eksiklerine odaklanır.

TODO(C): mock yerine llm.structured(..., schema=Plan) ile gerçek plan üret.
"""

from config import settings
from graph.state import Plan, ResearchState, SubQuestion


def run(state: ResearchState) -> dict:
    rnd = state.get("round", 0) + 1
    critique = state.get("critique")

    if critique and critique.gaps:
        topics = critique.gaps
    else:
        q = state["question"]
        topics = [f"{q} — tanım ve arka plan", f"{q} — güncel yaklaşımlar", f"{q} — sınırlamalar"]

    subs = [SubQuestion(id=f"r{rnd}q{i + 1}", text=t) for i, t in enumerate(topics)]
    return {"round": rnd, "plan": Plan(sub_questions=subs[: settings.max_subquestions])}
