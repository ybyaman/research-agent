import re

from graph.workflow import build_graph, run_config


def test_end_to_end_mock_runs_critic_loop():
    final = build_graph().invoke({"question": "Test sorusu", "round": 0}, config=run_config())

    # Mock Critic 1. turda eksik bildirir -> 2 tur çalışmalı
    assert final["round"] == 2
    assert final["critique"].verdict == "sufficient"

    # Her alt soru için kaynak bulundu ve hepsi okundu
    sub_ids = {s.sub_question_id for s in final["sources"]}
    assert {"r1q1", "r1q2", "r1q3", "r2q1"} <= sub_ids
    assert set(final["read_urls"]) == {s.url for s in final["sources"]}

    # Rapordaki her [n] bir kaynağa karşılık geliyor
    report = final["report"]
    used = {int(n) for n in re.findall(r"\[(\d+)\]", report.markdown)}
    assert used == {c.number for c in report.citations}
