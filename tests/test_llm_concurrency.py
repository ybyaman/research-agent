import threading
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest

from llm import client as client_mod
from llm.client import LLMClient, LLMError

OK = {"message": {"content": "cevap"}, "prompt_eval_count": 10, "eval_count": 5}


class SlowServer:
    """Her isteği biraz bekleten ve aynı anda kaç istek işlendiğini ölçen sahte Ollama."""

    def __init__(self, delay: float = 0.05):
        self.delay = delay
        self.active = 0
        self.peak = 0
        self.lock = threading.Lock()

    def __call__(self, request: httpx.Request) -> httpx.Response:
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
        time.sleep(self.delay)
        with self.lock:
            self.active -= 1
        return httpx.Response(200, json=OK)


def _client(handler, **kw) -> LLMClient:
    http = httpx.Client(base_url="http://test", transport=httpx.MockTransport(handler))
    return LLMClient(http=http, model="fake", backoff=0.0, **kw)


@pytest.mark.parametrize("limit", [1, 2, 3])
def test_parallel_calls_never_exceed_limit(limit):
    server = SlowServer()
    llm = _client(server, max_parallel=limit)
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda _: llm.chat("p"), range(8)))
    assert server.peak == limit


def test_stats_are_consistent_under_threads():
    llm = _client(SlowServer(delay=0.01), max_parallel=2)
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda _: llm.chat("p"), range(20)))
    stats = llm.stats_snapshot()
    assert stats["calls"] == 20
    assert stats["prompt_tokens"] == 200 and stats["output_tokens"] == 100
    assert stats["wait_seconds"] > 0  # 8 thread, 2 slot: biri mutlaka bekledi


def test_seconds_excludes_queue_wait():
    llm = _client(SlowServer(delay=0.05), max_parallel=1)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda _: llm.chat("p"), range(4)))
    stats = llm.stats_snapshot()
    # 4 çağrı x 0.05s sunucu süresi; sıra beklemesi dahil olsaydı ~0.5s olurdu
    assert 0.18 < stats["seconds"] < 0.35
    assert stats["wait_seconds"] > 0.2


def _sequence(*steps):
    """Sırayla hata fırlatan ya da cevap dönen handler."""
    queue = list(steps)
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        step = queue.pop(0)
        if isinstance(step, Exception):
            raise step
        return httpx.Response(step, json=OK if step == 200 else {"error": "x"})

    return handler, calls


def test_retries_connection_error_then_succeeds():
    handler, calls = _sequence(httpx.ConnectError("reddedildi"), 200)
    assert _client(handler, http_retries=2).chat("p") == "cevap"
    assert calls["n"] == 2


def test_retries_5xx_then_succeeds():
    handler, calls = _sequence(503, 500, 200)
    assert _client(handler, http_retries=2).chat("p") == "cevap"
    assert calls["n"] == 3


def test_gives_up_after_retries():
    handler, calls = _sequence(httpx.ReadTimeout("yavaş"), httpx.ReadTimeout("yavaş"), 200)
    llm = _client(handler, http_retries=1)
    with pytest.raises(LLMError, match="ReadTimeout"):
        llm.chat("p")
    assert calls["n"] == 2
    assert llm.stats_snapshot()["failed_calls"] == 1


def test_client_error_is_not_retried():
    handler, calls = _sequence(404, 200)  # ör. model adı yanlış
    with pytest.raises(LLMError, match="404"):
        _client(handler, http_retries=3).chat("p")
    assert calls["n"] == 1


def test_slot_is_released_after_failure():
    handler, _ = _sequence(404, 200)
    llm = _client(handler, max_parallel=1, http_retries=0)
    with pytest.raises(LLMError):
        llm.chat("p")
    assert llm.chat("p") == "cevap"  # slot sızsaydı burada sonsuza dek beklerdi


def test_get_llm_is_a_singleton_across_threads(monkeypatch):
    monkeypatch.setattr(client_mod, "_client", None)
    with ThreadPoolExecutor(max_workers=8) as pool:
        instances = list(pool.map(lambda _: client_mod.get_llm(), range(16)))
    assert len({id(i) for i in instances}) == 1
