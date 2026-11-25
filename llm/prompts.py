"""prompts/ klasöründeki prompt dosyalarını yükler. Promptlar koda gömülmez."""

from functools import cache
from pathlib import Path

PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"


@cache
def load_prompt(name: str) -> str:
    return (PROMPT_DIR / f"{name}.md").read_text(encoding="utf-8")


def render(name: str, **kwargs: str) -> str:
    return load_prompt(name).format(**kwargs)
