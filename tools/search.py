"""Arama araçları (Rol D). Hepsi ortak Source formatına dönüştürür.

Sözleşme: her fonksiyon list[Source] döndürür, hata durumunda boş liste + log (akışı kırmaz).
"""

from graph.state import Source


def web_search(query: str, sub_question_id: str, k: int = 4) -> list[Source]:
    """Tavily ile arar; anahtar yoksa veya kota dolduysa DuckDuckGo (ddgs) yedeğine düşer."""
    raise NotImplementedError("Rol D")


def arxiv_search(query: str, sub_question_id: str, k: int = 3) -> list[Source]:
    raise NotImplementedError("Rol D")


def semantic_scholar_search(query: str, sub_question_id: str, k: int = 3) -> list[Source]:
    raise NotImplementedError("Rol D")
