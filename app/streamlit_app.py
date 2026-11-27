"""Streamlit arayüzü (Rol E). Çalıştır: streamlit run app/streamlit_app.py

Arayüz yalnızca graph.runner.stream_research ve RunEvent şemasını kullanır.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st  # noqa: E402

from config import settings  # noqa: E402
from graph.runner import stream_research  # noqa: E402
from graph.state import Report  # noqa: E402

NODE_LABELS = {"planner": "Planner", "searcher": "Searcher", "reader": "Reader",
               "critic": "Critic", "writer": "Writer"}

st.set_page_config(page_title="Research Assistant", layout="wide")
st.title("Multi-Agent Research Assistant")
st.caption(f"Mod: {'mock' if settings.is_mock else 'real'} · Model: {settings.llm_model}")

question = st.text_input("Araştırma sorusu", placeholder="Örn. Küçük dil modelleri RAG'de ne kadar başarılı?")

if st.button("Araştır", type="primary", disabled=not question):
    report = None
    summary = {}
    # TODO(E): turlara göre gruplama, kaynak kartları, ilerleme çubuğu
    for ev in stream_research(question):
        if ev.kind == "node_finished":
            label = f"Tur {ev.round} · {NODE_LABELS.get(ev.node, ev.node)} — {ev.message}"
            with st.status(label, state="complete", expanded=False):
                st.json(ev.data, expanded=False)
        elif ev.kind == "node_error":
            st.warning(ev.message)
        elif ev.kind == "run_finished":
            summary = ev.data
            if ev.data["report"]:
                report = Report.model_validate(ev.data["report"])

    if summary:
        cols = st.columns(5)
        cols[0].metric("Tur", summary["rounds"])
        cols[1].metric("Kaynak", summary["source_count"])
        cols[2].metric("Bulgu", summary["fact_count"])
        cols[3].metric("LLM çağrısı", int(summary["llm"]["calls"]))
        cols[4].metric("Hata", summary["error_count"])

    if report:
        st.divider()
        st.markdown(report.markdown)
        st.download_button("Raporu indir (.md)", report.markdown, file_name="rapor.md")
