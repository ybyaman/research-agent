"""Bir ajan çökse bile akış durmamalı, hata kaydedilmeli ve yine bir rapor üretilmeli."""

import re

import pytest

from agents import critic, planner, reader, searcher, writer
from graph.state import Plan
from graph.workflow import build_graph, run_config


def _run() -> dict:
    return build_graph().invoke({"question": "Test sorusu", "round": 0}, config=run_config())


def _boom(*_args, **_kwargs):
    raise RuntimeError("patladı")


def _assert_valid_report(final: dict) -> None:
    report = final["report"]
    used = {int(n) for n in re.findall(r"\[(\d+)\]", report.markdown)}
    assert used == {c.number for c in report.citations}


def test_happy_path_has_no_errors():
    assert _run().get("errors", []) == []


def test_one_searcher_failing_keeps_others(monkeypatch):
    original = searcher.run

    def flaky(task):
        if task["sub_question"].id == "r1q2":
            raise TimeoutError("arama zaman aşımı")
        return original(task)

    monkeypatch.setattr(searcher, "run", flaky)
    final = _run()

    [err] = final["errors"]
    assert err.node == "searcher"
    assert err.sub_question_id == "r1q2"
    assert "TimeoutError" in err.message

    found = {s.sub_question_id for s in final["sources"]}
    assert "r1q2" not in found and {"r1q1", "r1q3"} <= found
    _assert_valid_report(final)


def test_planner_failure_falls_back_to_question(monkeypatch):
    monkeypatch.setattr(planner, "run", _boom)
    final = _run()

    assert {e.node for e in final["errors"]} == {"planner"}
    assert final["plan"].sub_questions[0].text == "Test sorusu"
    assert final["round"] == 2  # yedek plan da round'u artırır, döngü sınırı çalışır
    _assert_valid_report(final)


def test_reader_failure_still_produces_report(monkeypatch):
    monkeypatch.setattr(reader, "run", _boom)
    final = _run()

    assert {e.node for e in final["errors"]} == {"reader"}
    assert final.get("facts", []) == []
    assert "bulgu bulunamadı" in final["report"].markdown


def test_critic_failure_goes_to_writer(monkeypatch):
    monkeypatch.setattr(critic, "run", _boom)
    final = _run()

    assert final["round"] == 1  # Critic çökünce tekrar tura girilmedi
    assert final["critique"].verdict == "sufficient"
    assert [e.node for e in final["errors"]] == ["critic"]
    _assert_valid_report(final)


def test_writer_failure_gives_basic_report(monkeypatch):
    monkeypatch.setattr(writer, "run", _boom)
    final = _run()

    assert [e.node for e in final["errors"]] == ["writer"]
    assert "hata verdi" in final["report"].markdown
    assert final["report"].citations
    _assert_valid_report(final)


def test_empty_plan_skips_to_writer(monkeypatch):
    monkeypatch.setattr(planner, "run", lambda s: {"round": 1, "plan": Plan(sub_questions=[])})
    final = _run()

    assert final.get("sources", []) == []
    assert "bulgu bulunamadı" in final["report"].markdown


@pytest.mark.parametrize("module", [planner, reader, critic, writer])
def test_every_node_failing_never_raises(monkeypatch, module):
    monkeypatch.setattr(module, "run", _boom)
    assert _run()["report"].markdown
