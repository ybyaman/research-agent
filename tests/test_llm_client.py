import json
import os

import httpx
import pytest

from graph.state import Plan
from llm.client import LLMClient, LLMError


def _client(responses: list[str]) -> LLMClient:
    queue = list(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"message": {"content": queue.pop(0)}})

    http = httpx.Client(base_url="http://test", transport=httpx.MockTransport(handler))
    return LLMClient(http=http, model="fake")


VALID = json.dumps({"sub_questions": [{"id": "q1", "text": "a"}]})


def test_structured_parses_valid_json():
    plan = _client([VALID]).structured("p", Plan)
    assert plan.sub_questions[0].id == "q1"


def test_structured_retries_on_bad_output():
    plan = _client(["bozuk", VALID]).structured("p", Plan)
    assert len(plan.sub_questions) == 1


def test_structured_gives_up_after_retries():
    with pytest.raises(LLMError):
        _client(["x", "y", "z"]).structured("p", Plan, retries=2)


@pytest.mark.skipif(not os.getenv("RUN_OLLAMA_TESTS"), reason="RUN_OLLAMA_TESTS=1 ile çalışır")
def test_real_ollama_structured():
    plan = LLMClient().structured(
        "Şu soruyu 3 alt soruya böl: Güneş enerjisinin avantajları nelerdir?", Plan
    )
    assert 1 <= len(plan.sub_questions) <= 6
