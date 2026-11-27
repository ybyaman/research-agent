"""Test setini çalıştırır, raporları reports/ altına yazar ve basit metrikler basar (Rol E).

Çalıştır: python -m eval.run_eval
TODO(E): kaynak doğruluğu (atıflar gerçekten bulgularda mı), kapsam skoru, süre/token.
"""

import json
import time
from pathlib import Path

from graph.runner import run_research

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    questions = json.loads((ROOT / "eval" / "questions.json").read_text(encoding="utf-8"))
    out_dir = ROOT / "reports"
    out_dir.mkdir(exist_ok=True)
    for item in questions:
        t0 = time.perf_counter()
        state = run_research(item["question"])
        report = state["report"]
        (out_dir / f"{item['id']}.md").write_text(report.markdown, encoding="utf-8")
        print(f"{item['id']}: tur={state['round']} kaynak={len(state['sources'])} "
              f"bulgu={len(state['facts'])} atıf={len(report.citations)} "
              f"süre={time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
