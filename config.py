"""Tüm ayarların tek kaynağı. Değerler ortam değişkenlerinden veya .env dosyasından okunur."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


@dataclass(frozen=True)
class Settings:
    llm_mode: str = field(default_factory=lambda: _env("LLM_MODE", "mock"))  # "mock" | "real"
    ollama_host: str = field(default_factory=lambda: _env("OLLAMA_HOST", "http://localhost:11434"))
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", "qwen3:8b"))
    embed_model: str = field(default_factory=lambda: _env("EMBED_MODEL", "nomic-embed-text"))
    llm_timeout: int = field(default_factory=lambda: _int("LLM_TIMEOUT", 120))
    max_concurrency: int = field(default_factory=lambda: _int("MAX_CONCURRENCY", 4))  # graf node'ları
    llm_max_parallel: int = field(default_factory=lambda: _int("LLM_MAX_PARALLEL", 1))  # Ollama'ya eşzamanlı istek
    llm_http_retries: int = field(default_factory=lambda: _int("LLM_HTTP_RETRIES", 2))
    max_rounds: int = field(default_factory=lambda: _int("MAX_ROUNDS", 2))
    max_subquestions: int = field(default_factory=lambda: _int("MAX_SUBQUESTIONS", 5))
    sources_per_question: int = field(default_factory=lambda: _int("SOURCES_PER_QUESTION", 4))
    tavily_api_key: str = field(default_factory=lambda: _env("TAVILY_API_KEY", ""))
    log_dir: str = field(default_factory=lambda: _env("LOG_DIR", "logs"))

    @property
    def is_mock(self) -> bool:
        return self.llm_mode.lower() != "real"


settings = Settings()
