"""Ortak state ve veri şemaları — ajanlar arası SÖZLEŞME.

Bu dosya 1. haftanın sonunda dondurulur. Sonrasında değişiklik için
teknik lider (A) onayı ve PR gerekir.

Her ajan aynı imzayı kullanır:
    def run(state: ResearchState) -> dict
ve state'in yalnızca değiştirdiği alanları döndürür.
"""

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field

# --------------------------------------------------------------------------- #
# Veri şemaları
# --------------------------------------------------------------------------- #


class SubQuestion(BaseModel):
    id: str = Field(description="Kısa kimlik, örn. 'r1q1' (tur + sıra)")
    text: str = Field(description="Tek başına aranabilir alt soru")


class Plan(BaseModel):
    sub_questions: list[SubQuestion]


SourceKind = Literal["web", "arxiv", "semantic_scholar", "other"]


class Source(BaseModel):
    """Searcher'ın ürettiği, Reader'ın tükettiği kaynak."""

    url: str
    title: str
    snippet: str = ""
    kind: SourceKind = "web"
    sub_question_id: str = Field(description="Hangi alt soru için bulundu")


class Fact(BaseModel):
    """Reader'ın bir kaynaktan çıkardığı, kaynağa bağlı tek bilgi."""

    claim: str = Field(description="Tek cümlelik bilgi")
    source_url: str
    quote: str = Field("", description="İddiayı destekleyen kısa alıntı")
    sub_question_id: str
    confidence: float = Field(0.5, ge=0.0, le=1.0)


class Critique(BaseModel):
    """Critic kararı."""

    verdict: Literal["sufficient", "needs_more"]
    gaps: list[str] = Field(default_factory=list, description="Eksik kalan konular")
    notes: str = ""


class Citation(BaseModel):
    number: int
    url: str
    title: str


class Report(BaseModel):
    title: str
    markdown: str = Field(description="[1], [2] şeklinde atıflı rapor gövdesi")
    citations: list[Citation]


# --------------------------------------------------------------------------- #
# Graph state
# --------------------------------------------------------------------------- #


class ResearchState(TypedDict, total=False):
    question: str  # kullanıcı sorusu (giriş)
    round: int  # Planner her çağrıldığında 1 artar
    plan: Plan  # son turun planı
    sources: Annotated[list[Source], operator.add]  # paralel Searcher'lar ekler
    facts: Annotated[list[Fact], operator.add]  # Reader ekler
    read_urls: Annotated[list[str], operator.add]  # Reader'ın işlediği URL'ler
    critique: Critique
    report: Report


class SearchTask(TypedDict):
    """Paralel Searcher'a gönderilen tekil görev (LangGraph Send)."""

    question: str
    sub_question: SubQuestion
