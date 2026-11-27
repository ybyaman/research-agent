from agents import searcher
from graph.state import (
    NodeError,
    Source,
    merge_sources,
    merge_unique,
    normalize_url,
)
from graph.workflow import build_graph, run_config


def _src(url: str, sq: str = "q1") -> Source:
    return Source(url=url, title=url, sub_question_id=sq)


def test_normalize_url_strips_noise():
    a = normalize_url("HTTPS://Example.com/Page/?utm_source=x&id=3#bolum")
    b = normalize_url("https://example.com/Page?id=3")
    assert a == b


def test_normalize_url_keeps_real_params():
    assert normalize_url("https://x.com/a?refresh=1") != normalize_url("https://x.com/a")
    assert normalize_url("https://x.com/a?id=1") != normalize_url("https://x.com/a?id=2")


def test_merge_sources_skips_duplicates_first_wins():
    left = [_src("https://a.com/x", "q1")]
    right = [_src("https://a.com/x/", "q2"), _src("https://b.com", "q2"), _src("https://b.com", "q3")]
    merged = merge_sources(left, right)
    assert [s.url for s in merged] == ["https://a.com/x", "https://b.com"]
    assert merged[0].sub_question_id == "q1"


def test_merge_sources_handles_empty():
    assert merge_sources(None, [_src("https://a.com")])[0].url == "https://a.com"
    assert merge_sources([], None) == []


def test_merge_unique_keeps_order():
    assert merge_unique(["a", "b"], ["b", "c", "a", "d"]) == ["a", "b", "c", "d"]


def test_parallel_searchers_returning_same_url_are_deduped(monkeypatch):
    """Paralel Searcher'lar aynı URL'yi bulsa da state'te tek kaynak kalmalı."""

    def same_url_searcher(task):
        return {"sources": [_src("https://shared.com/page", task["sub_question"].id)]}

    monkeypatch.setattr(searcher, "run", same_url_searcher)
    final = build_graph().invoke({"question": "T", "round": 0}, config=run_config())

    assert [s.url for s in final["sources"]] == ["https://shared.com/page"]
    assert final["read_urls"] == ["https://shared.com/page"]
    assert len(final["facts"]) == 1  # Reader aynı kaynağı ikinci turda tekrar okumadı


def test_errors_field_accumulates():
    e = NodeError(node="searcher", message="timeout", round=1, sub_question_id="r1q2")
    assert e.model_dump()["node"] == "searcher"
