"""UI ve eval için çalışma olayları (events) — E ile SÖZLEŞME.

Her node bittiğinde bir RunEvent üretilir. UI yalnızca bu şemayı bilir,
LangGraph'ın iç çıktı formatını bilmek zorunda değildir.

Olay sırası:
    run_started → (node_finished | node_error)* → run_finished
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from graph.state import Critique, Fact, NodeError, Plan, Report, Source

EventKind = Literal["run_started", "node_finished", "node_error", "run_finished"]


class RunEvent(BaseModel):
    run_id: str
    seq: int = Field(description="Çalışma içinde 0'dan artan sıra numarası")
    kind: EventKind
    node: str | None = None  # planner | searcher | reader | critic | writer
    round: int = 0
    elapsed: float = Field(description="Çalışma başından beri geçen saniye")
    message: str = Field(description="UI'da doğrudan gösterilebilecek Türkçe özet")
    data: dict[str, Any] = Field(default_factory=dict, description="Node'a özel ayrıntılar (JSON uyumlu)")
    # Yalnızca run_finished'da dolu; Python içi kullanım içindir, JSON'a yazılmaz.
    final_state: Any = Field(default=None, exclude=True, repr=False)


# --------------------------------------------------------------------------- #
# Node çıktılarını özetleyen fonksiyonlar: (output, ctx) -> (message, data)
# ctx: çalışma boyunca biriken bağlam (ör. alt soru id -> metin)
# --------------------------------------------------------------------------- #


def summarize_planner(out: dict, ctx: dict) -> tuple[str, dict]:
    plan: Plan | None = out.get("plan")
    subs = plan.sub_questions if plan else []
    ctx["sub_questions"].update({sq.id: sq.text for sq in subs})
    return (f"{len(subs)} alt soru planlandı",
            {"sub_questions": [sq.model_dump() for sq in subs]})


def summarize_searcher(out: dict, ctx: dict) -> tuple[str, dict]:
    sources: list[Source] = out.get("sources", [])
    errors: list[NodeError] = out.get("errors", [])
    sq_id = sources[0].sub_question_id if sources else (errors[0].sub_question_id if errors else None)
    sq_text = ctx["sub_questions"].get(sq_id, sq_id or "?")
    return (f"“{sq_text}” için {len(sources)} kaynak bulundu",
            {"sub_question_id": sq_id,
             "sources": [{"title": s.title, "url": s.url, "kind": s.kind} for s in sources]})


def summarize_reader(out: dict, ctx: dict) -> tuple[str, dict]:
    facts: list[Fact] = out.get("facts", [])
    read = out.get("read_urls", [])
    return (f"{len(read)} kaynak okundu, {len(facts)} bulgu çıkarıldı",
            {"read_count": len(read), "facts": [f.model_dump() for f in facts]})


def summarize_critic(out: dict, ctx: dict) -> tuple[str, dict]:
    c: Critique | None = out.get("critique")
    if c is None:
        return "Değerlendirme yapılamadı", {}
    msg = "Bulgular yeterli" if c.verdict == "sufficient" else f"Eksik var: {len(c.gaps)} konu araştırılacak"
    return msg, c.model_dump()


def summarize_writer(out: dict, ctx: dict) -> tuple[str, dict]:
    r: Report | None = out.get("report")
    if r is None:
        return "Rapor yazılamadı", {}
    return f"Rapor yazıldı ({len(r.citations)} kaynak)", {"citation_count": len(r.citations)}


SUMMARIZERS = {
    "planner": summarize_planner,
    "searcher": summarize_searcher,
    "reader": summarize_reader,
    "critic": summarize_critic,
    "writer": summarize_writer,
}


def error_message(err: NodeError) -> str:
    return f"{err.node} hata verdi, yedek çıktıyla devam edildi: {err.message}"
