"""Writer (Rol C). Bulgulardan [n] atıflı Markdown rapor yazar.

TODO(C): gövdeyi llm ile yaz; atıf numaralarını burada deterministik üret ki model uydurmasın.
"""

from graph.state import Citation, ResearchState, Report


def run(state: ResearchState) -> dict:
    facts = state.get("facts", [])
    titles = {s.url: s.title for s in state.get("sources", [])}

    numbers: dict[str, int] = {}
    for f in facts:
        numbers.setdefault(f.source_url, len(numbers) + 1)

    lines = [f"# {state['question']}", ""]
    for f in facts:
        lines.append(f"- {f.claim} [{numbers[f.source_url]}]")
    lines += ["", "## Kaynaklar", ""]
    citations = [Citation(number=n, url=u, title=titles.get(u, u)) for u, n in numbers.items()]
    lines += [f"{c.number}. [{c.title}]({c.url})" for c in citations]

    return {"report": Report(title=state["question"], markdown="\n".join(lines), citations=citations)}
