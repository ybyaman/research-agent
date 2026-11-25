"""Reader/Extractor (Rol B). Yeni kaynakları okur, kaynak bilgisiyle Fact çıkarır.

TODO(B): tools/fetch.py + tools/pdf.py ile metni çek, chunk'la, tools/embeddings.py ile
alakalı parçaları seç, llm.structured ile Fact listesi çıkar.
"""

from graph.state import Fact, ResearchState


def run(state: ResearchState) -> dict:
    seen = set(state.get("read_urls", []))
    new_sources = [s for s in state.get("sources", []) if s.url not in seen]

    facts = [
        Fact(
            claim=f"{s.title} kaynağına göre örnek bir bulgu.",
            source_url=s.url,
            quote=s.snippet,
            sub_question_id=s.sub_question_id,
            confidence=0.5,
        )
        for s in new_sources
    ]
    return {"facts": facts, "read_urls": [s.url for s in new_sources]}
