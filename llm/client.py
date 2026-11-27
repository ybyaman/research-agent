"""Ollama wrapper. Ajanlar LLM'e YALNIZCA bu modül üzerinden erişir.

Kullanım:
    from llm.client import get_llm
    plan = get_llm().structured(prompt, schema=Plan, system="...")

Thread-safe: LangGraph paralel node'ları ayrı thread'lerde çalıştırır. Ollama'ya
aynı anda en fazla `LLM_MAX_PARALLEL` istek gider; fazlası sırada bekler.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from contextlib import contextmanager
from typing import Iterator, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from config import settings

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)

_TRANSIENT = (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout,
              httpx.RemoteProtocolError)


class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(
        self,
        host: str | None = None,
        model: str | None = None,
        embed_model: str | None = None,
        timeout: int | None = None,
        max_parallel: int | None = None,
        http_retries: int | None = None,
        backoff: float = 1.0,
        http: httpx.Client | None = None,
    ) -> None:
        self.model = model or settings.llm_model
        self.embed_model = embed_model or settings.embed_model
        self.http_retries = settings.llm_http_retries if http_retries is None else http_retries
        self.backoff = backoff
        self._http = http or httpx.Client(
            base_url=host or settings.ollama_host,
            timeout=timeout or settings.llm_timeout,
        )
        self._slots = threading.BoundedSemaphore(max_parallel or settings.llm_max_parallel)
        self._lock = threading.Lock()
        # UI / eval için kullanım istatistikleri. Doğrudan değil, stats_snapshot() ile okuyun.
        self.stats = {"calls": 0, "failed_calls": 0, "prompt_tokens": 0, "output_tokens": 0,
                      "seconds": 0.0, "wait_seconds": 0.0}

    # ------------------------------------------------------------------ #
    # Eşzamanlılık, yeniden deneme ve istatistik
    # ------------------------------------------------------------------ #

    @contextmanager
    def _slot(self) -> Iterator[None]:
        """Ollama'ya gidecek istek için sıra bekler; bekleme süresi istatistiğe eklenir."""
        t0 = time.perf_counter()
        self._slots.acquire()
        self._add(wait_seconds=time.perf_counter() - t0)
        try:
            yield
        finally:
            self._slots.release()

    def _add(self, **deltas: float) -> None:
        with self._lock:
            for k, v in deltas.items():
                self.stats[k] += v

    def stats_snapshot(self) -> dict:
        with self._lock:
            return dict(self.stats)

    def _post(self, path: str, body: dict) -> dict:
        """Geçici hatalarda (bağlantı, zaman aşımı, 5xx) üstel beklemeyle yeniden dener.

        `seconds` istatistiğine yalnızca isteğin sunucuda geçirdiği süre eklenir;
        sırada bekleme `wait_seconds`, yeniden deneme beklemeleri hiçbirine eklenmez.
        """
        for attempt in range(self.http_retries + 1):
            try:
                with self._slot():
                    t0 = time.perf_counter()
                    try:
                        resp = self._http.post(path, json=body)
                    finally:
                        self._add(seconds=time.perf_counter() - t0)
                if resp.status_code >= 500:
                    raise httpx.HTTPStatusError(f"{resp.status_code} sunucu hatası",
                                                request=resp.request, response=resp)
                resp.raise_for_status()
                return resp.json()
            except (*_TRANSIENT, httpx.HTTPStatusError) as e:
                is_5xx = isinstance(e, httpx.HTTPStatusError) and e.response.status_code >= 500
                retryable = isinstance(e, _TRANSIENT) or is_5xx
                if not retryable or attempt == self.http_retries:
                    self._add(failed_calls=1)
                    raise LLMError(f"Ollama isteği başarısız ({path}): {type(e).__name__}: {e}") from e
                wait = self.backoff * 2**attempt
                log.warning("Ollama geçici hata (%s), %.1fs sonra tekrar denenecek", type(e).__name__, wait)
                time.sleep(wait)
            except httpx.HTTPError as e:
                self._add(failed_calls=1)
                raise LLMError(f"Ollama isteği başarısız ({path}): {type(e).__name__}: {e}") from e
        raise AssertionError("ulaşılamaz")

    # ------------------------------------------------------------------ #
    # Genel API
    # ------------------------------------------------------------------ #

    def chat(self, prompt: str, system: str = "", *, json_schema: dict | None = None,
             temperature: float = 0.2) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        body: dict = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "think": False,  # qwen3 gibi modellerde düşünme çıktısını kapatır
            "options": {"temperature": temperature},
        }
        if json_schema is not None:
            body["format"] = json_schema

        data = self._post("/api/chat", body)
        self._add(calls=1, prompt_tokens=data.get("prompt_eval_count", 0),
                  output_tokens=data.get("eval_count", 0))
        log.info("LLM çağrısı: %s girdi / %s çıktı token",
                 data.get("prompt_eval_count", "?"), data.get("eval_count", "?"))
        return data["message"]["content"]

    def structured(self, prompt: str, schema: type[T], system: str = "", *,
                   retries: int = 2, temperature: float = 0.0) -> T:
        """Çıktıyı Pydantic şemasına zorlar. Bozuk JSON gelirse hatayı modele geri verip tekrar dener."""
        json_schema = schema.model_json_schema()
        last_err: Exception | None = None
        current_prompt = prompt
        for attempt in range(retries + 1):
            raw = self.chat(current_prompt, system, json_schema=json_schema, temperature=temperature)
            try:
                return schema.model_validate(json.loads(raw))
            except (json.JSONDecodeError, ValidationError) as e:
                last_err = e
                log.warning("Geçersiz yapılandırılmış çıktı (deneme %d): %s", attempt + 1, e)
                current_prompt = (
                    f"{prompt}\n\nÖnceki cevabın şemaya uymadı: {str(e)[:300]}\n"
                    "Yalnızca şemaya uyan geçerli JSON döndür."
                )
        raise LLMError(f"{schema.__name__} şemasına uygun çıktı alınamadı: {last_err}")

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._post("/api/embed", {"model": self.embed_model, "input": texts})["embeddings"]


_client: LLMClient | None = None
_client_lock = threading.Lock()


def get_llm() -> LLMClient:
    """Süreç genelinde tek istemci: eşzamanlılık sınırı ancak böyle herkes için geçerli olur."""
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = LLMClient()
    return _client
