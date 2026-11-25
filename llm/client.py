"""Ollama wrapper. Ajanlar LLM'e YALNIZCA bu modül üzerinden erişir.

Kullanım:
    from llm.client import get_llm
    plan = get_llm().structured(prompt, schema=Plan, system="...")
"""

from __future__ import annotations

import json
import logging
import time
from typing import TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from config import settings

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(
        self,
        host: str | None = None,
        model: str | None = None,
        embed_model: str | None = None,
        timeout: int | None = None,
        http: httpx.Client | None = None,
    ) -> None:
        self.model = model or settings.llm_model
        self.embed_model = embed_model or settings.embed_model
        self._http = http or httpx.Client(
            base_url=host or settings.ollama_host,
            timeout=timeout or settings.llm_timeout,
        )
        # Basit kullanım istatistikleri (UI / eval için)
        self.stats = {"calls": 0, "prompt_tokens": 0, "output_tokens": 0, "seconds": 0.0}

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

        t0 = time.perf_counter()
        try:
            resp = self._http.post("/api/chat", json=body)
            resp.raise_for_status()
        except httpx.HTTPError as e:
            raise LLMError(f"Ollama isteği başarısız: {e}") from e
        data = resp.json()

        self.stats["calls"] += 1
        self.stats["prompt_tokens"] += data.get("prompt_eval_count", 0)
        self.stats["output_tokens"] += data.get("eval_count", 0)
        self.stats["seconds"] += time.perf_counter() - t0
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
        try:
            resp = self._http.post("/api/embed", json={"model": self.embed_model, "input": texts})
            resp.raise_for_status()
        except httpx.HTTPError as e:
            raise LLMError(f"Embedding isteği başarısız: {e}") from e
        return resp.json()["embeddings"]


_client: LLMClient | None = None


def get_llm() -> LLMClient:
    global _client
    if _client is None:
        _client = LLMClient()
    return _client
