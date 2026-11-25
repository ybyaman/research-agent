"""Streamlit arayüzü (Rol E). Çalıştır: streamlit run app/streamlit_app.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st  # noqa: E402

from config import settings  # noqa: E402
from graph.trace import start_run  # noqa: E402
from graph.workflow import build_graph, run_config  # noqa: E402

st.set_page_config(page_title="Research Assistant", layout="wide")
st.title("Multi-Agent Research Assistant")
st.caption(f"Mod: {'mock' if settings.is_mock else 'real'} · Model: {settings.llm_model}")

question = st.text_input("Araştırma sorusu", placeholder="Örn. Küçük dil modelleri RAG'de ne kadar başarılı?")

if st.button("Araştır", type="primary", disabled=not question):
    start_run()
    report = None
    # TODO(E): adımları daha zengin göster (kaynak listesi, bulgu sayısı, süre)
    for update in build_graph().stream({"question": question, "round": 0},
                                       config=run_config(), stream_mode="updates"):
        for node, out in update.items():
            with st.status(node, state="complete"):
                st.write({k: (len(v) if isinstance(v, list) else v) for k, v in (out or {}).items()})
            if out and "report" in out:
                report = out["report"]

    if report:
        st.divider()
        st.markdown(report.markdown)
        st.download_button("Raporu indir (.md)", report.markdown, file_name="rapor.md")
