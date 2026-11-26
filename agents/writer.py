"""Writer (Rol C). Bulgulardan [n] atıflı Markdown rapor yazar.

TODO(C): gövdeyi llm ile yaz; atıf numaralarını graph.report.number_sources ile
deterministik üret ki model uydurmasın.
"""

from graph.report import basic_report
from graph.state import ResearchState


def run(state: ResearchState) -> dict:
    return {"report": basic_report(state["question"], state.get("facts", []), state.get("sources", []))}
