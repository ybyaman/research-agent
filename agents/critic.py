"""Critic (Rol A). Bulguların soruyu cevaplamaya yetip yetmediğine karar verir.

İki katman:
1. Deterministik kapsam kontrolü: plandaki hiç bulgusu olmayan alt sorular kesin eksiktir.
2. LLM değerlendirmesi: bulgular soruyu cevaplıyor mu, çelişki var mı?

LLM hata verirse istisna yukarı fırlar; graph/resilience.py bunu yakalar ve
"sufficient" kabul ederek Writer'a geçer.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from config import settings
from graph.state import Critique, Fact, ResearchState, SubQuestion
from llm.client import get_llm
from llm.prompts import render

SYSTEM = (
    "Sen titiz ama pratik bir araştırma eleştirmenisin. "
    "Kararını yalnızca verilen bulgulara dayandırırsın ve gereksiz yere ek araştırma istemezsin."
)
MAX_FACTS_PER_SUBQ = 8  # küçük modelin context'ini korumak için
MAX_CLAIM_CHARS = 240
MAX_EXTRA_GAPS = 2  # model alt sorular dışında en fazla bu kadar yeni konu önerebilir


# --------------------------------------------------------------------------- #
# LLM'in doldurduğu iç şema. Küçük modeller tek bir genel kararda eksikleri
# kaçırıyor; önce her alt soruyu ayrı değerlendirtip kararı kodda veriyoruz.
# Bu şema yalnızca Critic'e ait; state sözleşmesi (Critique) değişmez.
# --------------------------------------------------------------------------- #


class SubAssessment(BaseModel):
    sub_question: str
    answered: bool = Field(description="Bulgular bu alt soruyu somut olarak cevaplıyor mu?")
    missing: str = Field("", description="Cevaplanmadıysa eksik olan şeyin kısa arama ifadesi")


class CriticAssessment(BaseModel):
    assessments: list[SubAssessment]
    question_answered: bool = Field(description="Bulgularla kullanıcının asıl sorusuna net cevap verilebilir mi?")
    extra_gaps: list[str] = Field(default_factory=list,
                                  description="Asıl soru için gerekli ama hiçbir alt soruda olmayan konular")
    notes: str = Field("", description="Çelişki veya tek kaynağa dayanan önemli iddia")

    def to_critique(self) -> Critique:
        gaps = [a.missing or a.sub_question for a in self.assessments if not a.answered]
        if not self.question_answered:
            gaps += self.extra_gaps[:MAX_EXTRA_GAPS]
        return Critique(verdict="needs_more" if gaps else "sufficient", gaps=gaps, notes=self.notes)


def run(state: ResearchState) -> dict:
    if settings.is_mock:
        return {"critique": _mock(state)}

    rnd = state.get("round", 0)
    if rnd >= settings.max_rounds:
        # Router zaten Writer'a gidecek; LLM çağrısı boşa olur.
        return {"critique": Critique(verdict="sufficient", notes="Tur sınırına ulaşıldı.")}

    subs = state["plan"].sub_questions
    facts = state.get("facts", [])
    uncovered = [sq.text for sq in subs if not _facts_for(sq, facts)]

    assessment = get_llm().structured(
        render(
            "critic",
            question=state["question"],
            coverage=_format_coverage(subs, facts),
            max_gaps=str(settings.max_subquestions),
        ),
        schema=CriticAssessment,
        system=SYSTEM,
    )
    llm_critique = drop_restated_gaps(assessment.to_critique(), facts)
    return {"critique": combine(llm_critique, uncovered, settings.max_subquestions)}


# --------------------------------------------------------------------------- #
# Yardımcılar (test edilebilsin diye ayrı)
# --------------------------------------------------------------------------- #


def _facts_for(sq: SubQuestion, facts: list[Fact]) -> list[Fact]:
    return [f for f in facts if f.sub_question_id == sq.id]


def _format_coverage(subs: list[SubQuestion], facts: list[Fact]) -> str:
    blocks = []
    for sq in subs:
        own = _facts_for(sq, facts)
        lines = [f"### {sq.text} ({len(own)} bulgu)"]
        if not own:
            lines.append("- (bulgu yok)")
        for f in own[:MAX_FACTS_PER_SUBQ]:
            lines.append(f"- {f.claim[:MAX_CLAIM_CHARS]}")
        if len(own) > MAX_FACTS_PER_SUBQ:
            lines.append(f"- ... ve {len(own) - MAX_FACTS_PER_SUBQ} bulgu daha")
        blocks.append("\n".join(lines))

    # Önceki turlardan kalan (bu planda olmayan) bulgular da kararı etkiler
    current = {sq.id for sq in subs}
    earlier = [f for f in facts if f.sub_question_id not in current]
    if earlier:
        lines = [f"### Önceki turlardan bulgular ({len(earlier)} bulgu)"]
        lines += [f"- {f.claim[:MAX_CLAIM_CHARS]}" for f in earlier[:MAX_FACTS_PER_SUBQ]]
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def drop_restated_gaps(critique: Critique, facts: list[Fact]) -> Critique:
    """Küçük modeller bazen bir bulguyu kopyalayıp 'eksik' diye yazar; bunları ele."""
    claims = [f.claim.strip().rstrip(".").lower() for f in facts if f.claim.strip()]
    gaps = [g for g in critique.gaps if not any(c in g.lower() for c in claims)]
    return critique.model_copy(update={"gaps": gaps})


def combine(llm: Critique, uncovered: list[str], max_gaps: int) -> Critique:
    """Deterministik ve LLM sonuçlarını birleştirir.

    - Bulgusuz alt soru varsa LLM ne derse desin needs_more.
    - needs_more dendi ama aranacak eksik yoksa Planner'ın yapacağı iş kalmaz -> sufficient.
    """
    gaps: list[str] = []
    seen: set[str] = set()
    llm_gaps = llm.gaps if llm.verdict == "needs_more" else []
    for g in [*uncovered, *llm_gaps]:
        g = g.strip()
        if g and g.lower() not in seen:
            seen.add(g.lower())
            gaps.append(g)
    gaps = gaps[:max_gaps]

    notes = llm.notes.strip()
    if uncovered:
        notes = (f"{len(uncovered)} alt soru için hiç bulgu yok. " + notes).strip()
    return Critique(verdict="needs_more" if gaps else "sufficient", gaps=gaps, notes=notes)


def _mock(state: ResearchState) -> Critique:
    """LLM'siz mod: 1. turda bir eksik bildirir (geri döngüyü test etmek için), sonra yeterli der."""
    if state.get("round", 0) == 1:
        return Critique(
            verdict="needs_more",
            gaps=[f"{state['question']} — farklı yaklaşımların karşılaştırması"],
            notes="Mock: karşılaştırma eksik.",
        )
    return Critique(verdict="sufficient", notes="Mock: yeterli.")
