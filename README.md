# Multi-Agent Research Assistant

Kullanıcının sorusunu alt sorulara bölen, web ve akademik kaynaklarda arayan, bulguları doğrulayan
ve kaynak gösterimli rapor yazan, **lokal LLM (Ollama)** ile çalışan çok ajanlı araştırma asistanı.

```
Soru → Planner → Searcher (paralel) → Reader → Critic ─┬─ yeterli → Writer → Rapor
          ▲                                             │
          └──────────── eksik var (en fazla MAX_ROUNDS) ┘
```

## Kurulum

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[dev]"          # rolüne göre: ".[dev,search]" veya ".[dev,reader]"
cp .env.example .env
```

Gerçek modelle çalışmak için Ollama gerekir:

```bash
ollama pull qwen3:8b
ollama pull nomic-embed-text
```

`.env` içinde `LLM_MODE=real` yap. Ortak sunucu kullanılıyorsa `OLLAMA_HOST=http://<ip>:11434`.

## Çalıştırma

```bash
pytest                                   # mock modda uçtan uca test
python -m graph.runner "Sorun buraya"    # terminalden, adımlarıyla
streamlit run app/streamlit_app.py       # arayüz
python -m eval.run_eval                  # test seti
```

## UI ve eval için event API

Akış `graph.runner.stream_research(question)` ile adım adım izlenir; her adım bir `RunEvent` döner
(şema: `graph/events.py`).

| `kind` | Ne zaman | `data` |
| --- | --- | --- |
| `run_started` | Başta, bir kez | `question`, `mode`, `model` |
| `node_finished` | Her node bittiğinde | Node'a özel: alt sorular, kaynaklar, bulgular, Critic kararı… |
| `node_error` | Node hata verip yedek çıktıyla devam ettiğinde | `NodeError` alanları |
| `run_finished` | Sonda, bir kez | `report`, `rounds`, `source_count`, `fact_count`, `error_count`, `llm` (çağrı, token, süre) |

`message` alanı UI'da doğrudan gösterilebilecek Türkçe özettir. Yalnızca sonucu isteyenler
`run_research(question)` kullanır.

## Klasörler ve sahipleri

| Yol | İçerik | Rol |
| --- | --- | --- |
| `config.py`, `llm/`, `graph/` | Ayarlar, Ollama wrapper, state, LangGraph akışı, trace | A |
| `agents/critic.py` | Critic ajanı | A |
| `agents/reader.py`, `tools/fetch.py`, `tools/pdf.py`, `tools/embeddings.py` | Okuma ve bilgi çıkarma | B |
| `agents/planner.py`, `agents/writer.py`, `prompts/` | Plan ve rapor | C |
| `agents/searcher.py`, `tools/search.py` | Arama entegrasyonları | D |
| `app/`, `eval/` | Arayüz, değerlendirme, sunum | E |

## Sözleşme kuralları

1. Her ajan `def run(state: ResearchState) -> dict` imzasını kullanır ve yalnızca değiştirdiği alanları döndürür
   (Searcher istisna: `SearchTask` alır).
2. `graph/state.py` 1. hafta sonunda dondurulur; değişiklik için A onayı + PR.
3. Şu an tüm ajanlar **mock**. Herkes kendi ajanını gerçeğiyle değiştirir, diğerlerini beklemez.
4. LLM'e yalnızca `llm/client.py` üzerinden erişilir (`get_llm().structured(...)`).
5. Promptlar `prompts/*.md` dosyalarında tutulur, koda gömülmez.

## Git akışı

- `main` korunur; herkes `rol/<kisa-aciklama>` branch'inde çalışır ve PR açar, A inceler.
- PR'lar `pytest`'ten geçmeli (mock mod, Ollama gerekmez).
