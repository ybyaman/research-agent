"""Her ajan çağrısını logs/<run_id>.jsonl dosyasına yazar (UI ve eval bunu okur)."""

from __future__ import annotations

import json
import time
import uuid
from functools import wraps
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel

from config import settings

_run_file: Path | None = None


def start_run() -> str:
    global _run_file
    run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    Path(settings.log_dir).mkdir(parents=True, exist_ok=True)
    _run_file = Path(settings.log_dir) / f"{run_id}.jsonl"
    return run_id


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, BaseModel):
        return obj.model_dump()
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def traced(name: str, fn: Callable[[Any], dict]) -> Callable[[Any], dict]:
    @wraps(fn)
    def wrapper(state: Any) -> dict:
        t0 = time.perf_counter()
        out = fn(state)
        if _run_file is not None:
            rec = {"ts": time.time(), "node": name,
                   "seconds": round(time.perf_counter() - t0, 3), "output": _jsonable(out)}
            with _run_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return out

    return wrapper
