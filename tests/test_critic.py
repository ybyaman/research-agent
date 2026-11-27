import dataclasses
import os

import pytest

from agents import critic
from agents.critic import CriticAssessment, SubAssessment
from config import settings
from graph.state import Fact, Plan, SubQuestion
from llm.client import LLMError


class FakeLLM:
    def __init__(self, answer: CriticAssessment | Exception):
        self.answer = answer
        self.prompts: list[str] = []

    def structured(self, prompt, schema, system=""):
        self.prompts.append(prompt)
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


@pytest.fixture
def real_mode(monkeypatch):
    monkeypatch.setattr(critic, "settings", dataclasses.replace(settings, llm_mode="real", max_rounds=2,
                                                                max_subquestions=3))


def _use(monkeypatch, answer) -> FakeLLM:
    fake = FakeLLM(answer)
    monkeypatch.setattr(critic, "get_llm", lambda: fake)
    return fake


def _assess(q1=True, q2=True, missing2="", answered=True, extra=()) -> CriticAssessment:
    return CriticAssessment(
        assessments=[SubAssessment(sub_question="RAG nedir", answered=q1),
                     SubAssessment(sub_question="RAG maliyeti", answered=q2, missing=missing2)],
        question_answered=answered, extra_gaps=list(extra),
    )


def _state(round_=1, fact_ids=("q1", "q2")) -> dict:
    subs = [SubQuestion(id="q1", text="RAG nedir"), SubQuestion(id="q2", text="RAG maliyeti")]
    facts = [Fact(claim=f"{i} hakkında bulgu", source_url=f"https://x.com/{i}", sub_question_id=i)
             for i in fact_ids]
    return {"question": "RAG kullanmalı mıyım?", "round": round_, "plan": Plan(sub_questions=subs),
            "facts": facts}


def test_sufficient_when_llm_agrees_and_all_covered(real_mode, monkeypatch):
    fake = _use(monkeypatch, _assess())
    out = critic.run(_state())["critique"]
    assert out.verdict == "sufficient" and out.gaps == []
    # Prompt bulguları ve alt soruları içeriyor
    assert "q1 hakkında bulgu" in fake.prompts[0] and "RAG maliyeti (1 bulgu)" in fake.prompts[0]


def test_uncovered_subquestion_forces_needs_more(real_mode, monkeypatch):
    _use(monkeypatch, _assess())  # LLM "her şey tamam" dese bile
    out = critic.run(_state(fact_ids=("q1",)))["critique"]
    assert out.verdict == "needs_more"
    assert out.gaps == ["RAG maliyeti"]
    assert "1 alt soru için hiç bulgu yok" in out.notes


def test_unanswered_subquestion_uses_missing_as_gap(real_mode, monkeypatch):
    _use(monkeypatch, _assess(q2=False, missing2="RAG altyapı maliyeti dolar"))
    out = critic.run(_state())["critique"]
    assert out.verdict == "needs_more" and out.gaps == ["RAG altyapı maliyeti dolar"]


def test_unanswered_without_missing_falls_back_to_subquestion(real_mode, monkeypatch):
    _use(monkeypatch, _assess(q2=False))
    assert critic.run(_state())["critique"].gaps == ["RAG maliyeti"]


def test_extra_gaps_only_when_question_unanswered(real_mode, monkeypatch):
    _use(monkeypatch, _assess(answered=True, extra=["gereksiz konu"]))
    assert critic.run(_state())["critique"].gaps == []

    _use(monkeypatch, _assess(answered=False, extra=["güvenlik", "lisans", "üçüncü konu"]))
    out = critic.run(_state())["critique"]
    assert out.gaps == ["güvenlik", "lisans"]  # MAX_EXTRA_GAPS = 2


def test_gaps_are_merged_deduped_and_capped(real_mode, monkeypatch):
    _use(monkeypatch, _assess(q2=False, missing2="rag maliyeti", answered=False,
                              extra=[" güvenlik ", "ölçeklenebilirlik"]))
    out = critic.run(_state(fact_ids=("q1",)))["critique"]
    assert out.gaps == ["RAG maliyeti", "güvenlik", "ölçeklenebilirlik"]  # max_subquestions=3


def test_gap_restating_a_fact_is_dropped(real_mode, monkeypatch):
    _use(monkeypatch, _assess(q2=False, missing2="q2 hakkında bulgu (bölge bazlı)"))
    assert critic.run(_state())["critique"].verdict == "sufficient"


def test_last_round_skips_llm(real_mode, monkeypatch):
    fake = _use(monkeypatch, _assess(answered=False, extra=["x"]))
    out = critic.run(_state(round_=2))["critique"]
    assert out.verdict == "sufficient" and fake.prompts == []


def test_long_fact_lists_are_truncated(real_mode, monkeypatch):
    fake = _use(monkeypatch, _assess())
    state = _state()
    state["facts"] = [Fact(claim="uzun " * 100, source_url=f"https://x.com/{n}", sub_question_id="q1")
                      for n in range(20)] + state["facts"]
    critic.run(state)
    assert f"... ve {20 + 1 - critic.MAX_FACTS_PER_SUBQ} bulgu daha" in fake.prompts[0]


def test_earlier_round_facts_are_shown(real_mode, monkeypatch):
    fake = _use(monkeypatch, _assess())
    state = _state()
    state["facts"].append(Fact(claim="eski bulgu", source_url="https://x.com/old", sub_question_id="r0q9"))
    critic.run(state)
    assert "Önceki turlardan bulgular (1 bulgu)" in fake.prompts[0]


def test_llm_error_propagates_for_resilience_layer(real_mode, monkeypatch):
    _use(monkeypatch, LLMError("bozuk"))
    with pytest.raises(LLMError):
        critic.run(_state())


@pytest.mark.skipif(not os.getenv("RUN_OLLAMA_TESTS"), reason="RUN_OLLAMA_TESTS=1 ile çalışır")
def test_real_ollama_critic(real_mode):
    out = critic.run(_state())["critique"]
    assert out.verdict in {"sufficient", "needs_more"}
    assert len(out.gaps) <= 3
