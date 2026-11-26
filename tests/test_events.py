import json

from agents import searcher
from graph.events import RunEvent
from graph.runner import run_research, stream_research
from graph.state import Report


def _events(question="Test sorusu") -> list[RunEvent]:
    return list(stream_research(question))


def test_event_order_and_sequence():
    evs = _events()
    assert evs[0].kind == "run_started" and evs[-1].kind == "run_finished"
    assert [e.seq for e in evs] == list(range(len(evs)))
    assert all(a.elapsed <= b.elapsed for a, b in zip(evs, evs[1:]))
    assert len({e.run_id for e in evs}) == 1

    nodes = [e.node for e in evs if e.kind == "node_finished"]
    # Mock: 1. turda 3 alt soru, 2. turda 1 alt soru
    assert nodes == ["planner", *["searcher"] * 3, "reader", "critic",
                     "planner", "searcher", "reader", "critic", "writer"]


def test_round_is_tracked():
    rounds = [(e.node, e.round) for e in _events() if e.kind == "node_finished"]
    assert rounds[0] == ("planner", 1)
    assert ("critic", 1) in rounds and ("critic", 2) in rounds
    assert rounds[-1] == ("writer", 2)


def test_messages_are_human_readable():
    msgs = {e.node: e.message for e in _events() if e.kind == "node_finished"}
    assert "alt soru planlandı" in msgs["planner"]
    assert "kaynak bulundu" in msgs["searcher"]
    assert "bulgu çıkarıldı" in msgs["reader"]
    assert msgs["writer"].startswith("Rapor yazıldı")


def test_searcher_message_names_its_subquestion():
    searcher_msgs = [e.message for e in _events("Güneş") if e.node == "searcher"]
    assert "“Güneş — tanım ve arka plan” için 2 kaynak bulundu" in searcher_msgs


def test_critic_event_carries_decision():
    critics = [e for e in _events() if e.node == "critic"]
    assert critics[0].data["verdict"] == "needs_more" and critics[0].data["gaps"]
    assert critics[0].message.startswith("Eksik var")
    assert critics[1].data["verdict"] == "sufficient"


def test_run_finished_has_report_and_counts():
    last = _events()[-1]
    report = Report.model_validate(last.data["report"])
    assert report.citations
    assert last.data["rounds"] == 2
    assert last.data["source_count"] == 8 and last.data["fact_count"] == 8
    assert last.data["error_count"] == 0
    assert set(last.data["llm"]) >= {"calls", "seconds", "wait_seconds"}


def test_events_are_json_serializable_without_state():
    for ev in _events():
        dumped = json.loads(ev.model_dump_json())
        assert "final_state" not in dumped


def test_node_error_event(monkeypatch):
    original = searcher.run

    def flaky(task):
        if task["sub_question"].id == "r1q2":
            raise TimeoutError("arama zaman aşımı")
        return original(task)

    monkeypatch.setattr(searcher, "run", flaky)
    evs = _events()
    [err] = [e for e in evs if e.kind == "node_error"]
    assert err.node == "searcher" and err.data["sub_question_id"] == "r1q2"
    assert "TimeoutError" in err.message
    # Hata olayı, aynı node'un node_finished olayından hemen sonra gelir
    prev = evs[err.seq - 1]
    assert prev.kind == "node_finished" and prev.node == "searcher" and "0 kaynak" in prev.message
    assert evs[-1].data["error_count"] == 1


def test_run_research_returns_final_state():
    final = run_research("Test sorusu")
    assert final["round"] == 2 and final["report"].citations
