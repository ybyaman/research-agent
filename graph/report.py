"""Deterministik rapor yardımcıları.

Atıf numaraları LLM'e bırakılmaz, burada üretilir; böylece model kaynak uyduramaz.
Writer (C) bunları kullanabilir; Writer çökerse yedek rapor da buradan gelir.
"""

from __future__ import annotations

from graph.state import Citation, Fact, Report, Source


def number_sources(facts: list[Fact]) -> dict[str, int]:
    """Bulgularda geçen her URL'ye ilk görülme sırasına göre 1'den başlayan numara verir."""
    numbers: dict[str, int] = {}
    for f in facts:
        numbers.setdefault(f.source_url, len(numbers) + 1)
    return numbers


def build_citations(numbers: dict[str, int], sources: list[Source]) -> list[Citation]:
    titles = {s.url: s.title for s in sources}
    return [Citation(number=n, url=u, title=titles.get(u, u)) for u, n in numbers.items()]


def basic_report(question: str, facts: list[Fact], sources: list[Source], note: str = "") -> Report:
    """LLM kullanmadan, bulguları madde madde listeleyen atıflı rapor."""
    lines = [f"# {question}", ""]
    if note:
        lines += [f"> {note}", ""]

    if not facts:
        lines.append("Bu soru için kullanılabilir bir bulgu bulunamadı.")
        return Report(title=question, markdown="\n".join(lines), citations=[])

    numbers = number_sources(facts)
    lines += [f"- {f.claim} [{numbers[f.source_url]}]" for f in facts]
    citations = build_citations(numbers, sources)
    lines += ["", "## Kaynaklar", ""]
    lines += [f"{c.number}. [{c.title}]({c.url})" for c in citations]
    return Report(title=question, markdown="\n".join(lines), citations=citations)
