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
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

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


class NodeError(BaseModel):
    """Bir node'da yakalanan ve akışı durdurmayan hata."""

    node: str = Field(description="Hatanın oluştuğu node, örn. 'searcher'")
    message: str
    round: int = 0
    sub_question_id: str | None = None


# --------------------------------------------------------------------------- #
# Reducer'lar: paralel ya da ardışık node çıktılarının state'e nasıl ekleneceği
# --------------------------------------------------------------------------- #

_TRACKING_PARAMS = {"fbclid", "gclid", "ref"}


def _is_tracking(key: str) -> bool:
    key = key.lower()
    return key.startswith("utm_") or key in _TRACKING_PARAMS


def normalize_url(url: str) -> str:
    """Aynı sayfanın farklı yazımlarını tek bir anahtara indirger (karşılaştırma içindir)."""
    parts = urlsplit(url.strip())
    query = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True) if not _is_tracking(k)]
    path = parts.path.rstrip("/") or "/"
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, urlencode(query), ""))


def merge_sources(left: list[Source] | None, right: list[Source] | None) -> list[Source]:
    """Yeni kaynakları ekler; normalize edilmiş URL'si zaten olanları atlar (ilk gelen kalır)."""
    merged = list(left or [])
    seen = {normalize_url(s.url) for s in merged}
    for s in right or []:
        key = normalize_url(s.url)
        if key not in seen:
            seen.add(key)
            merged.append(s)
    return merged


def merge_unique(left: list[str] | None, right: list[str] | None) -> list[str]:
    """Sırayı koruyarak tekrarsız birleştirir."""
    merged = list(left or [])
    seen = set(merged)
    for item in right or []:
        if item not in seen:
            seen.add(item)
            merged.append(item)
    return merged


# --------------------------------------------------------------------------- #
# Graph state
# --------------------------------------------------------------------------- #


class ResearchState(TypedDict, total=False):
    question: str  # kullanıcı sorusu (giriş)
    round: int  # Planner her çağrıldığında 1 artar
    plan: Plan  # son turun planı
    sources: Annotated[list[Source], merge_sources]  # paralel Searcher'lar ekler, URL'ye göre tekrarsız
    facts: Annotated[list[Fact], operator.add]  # Reader ekler
    read_urls: Annotated[list[str], merge_unique]  # Reader'ın işlediği URL'ler
    critique: Critique
    report: Report
    errors: Annotated[list[NodeError], operator.add]  # akışı durdurmayan hatalar


class SearchTask(TypedDict):
    """Paralel Searcher'a gönderilen tekil görev (LangGraph Send)."""

    question: str
    sub_question: SubQuestion
