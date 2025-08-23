# ui_streamlit.py
# Small Streamlit front-end for the CLI logic.
# Author: Jaxon Archer <Jaxon.Archer.MBA.MSF@gmail.com>

import os
import io
import csv
from typing import List, Dict, Tuple

# 1) Load .env and set defaults FIRST (so imports see them)
from dotenv import load_dotenv
load_dotenv()
os.environ.setdefault("UNIVERSE_LIMIT", "40")
os.environ.setdefault("OFFLINE_REC_FALLBACK", "1")
os.environ.setdefault("OSI_API_KEY", "demo")  # safe default for OSI

# 2) Imports that may read env at import-time
import streamlit as st
try:
    import pandas as pd
except Exception:
    pd = None  # optional

from agent.planner import build_recommendations, draft_email_from_recs
from agent.tools import ensure_sp500_universe
from app import answer_question
from agent.llm_factory import get_llm_callable
import app as appmod
import time

# Ensure LLM is initialized when app.py is imported by Streamlit
if not hasattr(appmod, "LLM") or not callable(appmod.LLM):
    appmod.LLM = get_llm_callable()

# -------- helpers --------
def _rows_to_csv(rows: List[Dict]) -> str:
    """Simple CSV writer used if pandas isn't available."""
    if not rows:
        return ""
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()

def _fmt_scores(rows: List[Dict]) -> List[Dict]:
    """Format score fields uniformly for display when pandas isn't present."""
    out = []
    for r in rows:
        rr = dict(r)
        for k in ("risk_score", "values_score", "composite"):
            if k in rr and isinstance(rr[k], (int, float)):
                rr[k] = f"{rr[k]:.3f}"
        out.append(rr)
    return out

# -------- page --------
st.set_page_config(page_title="Ethic Agent", page_icon="📊", layout="centered")
st.title("Ethic Agent — Candidate Exercise")
st.caption("Public-company analysis: fundamentals, sustainability signals, and filings. Vertex-first config.")

# Sidebar: quick sanity so reviewers see why output might be empty
with st.sidebar:
    provider = (os.getenv("LLM_PROVIDER") or "vertex").lower()
    st.subheader("Status")

    def status(var: str) -> str:
        return "set" if os.getenv(var) else "missing"

    st.text(f"LLM_PROVIDER: {provider}")
    st.text(f"GCP_PROJECT: {status('GCP_PROJECT')}")
    st.text(f"ALPHAVANTAGE_API_KEY: {status('ALPHAVANTAGE_API_KEY')}")
    st.text(f"OSI_API_KEY: {status('OSI_API_KEY')}")
    st.caption("Numbers come from tools; the model only polishes wording.")

@st.cache_data(ttl=60 * 60, show_spinner=False)  # hourly refresh is fine here
def _cached_universe():
    return ensure_sp500_universe()

@st.cache_data(ttl=5 * 60, show_spinner=True)    # short TTL to avoid rate limits
def _cached_recs(risk: str, values_tuple: Tuple[str, ...], k: int, explain: bool, nonce: int):
    # nonce is only to bust the cache when "Force refresh" is checked
    _ = _cached_universe()
    return build_recommendations(risk=risk, values=values_tuple, k=k, explain=explain)

tab_chat, tab_recs = st.tabs(["Chat", "Recommend"])

with tab_chat:
    with st.form("chat_form", clear_on_submit=False):
        q = st.text_area("Your question", value="Tell me about MSFT's sustainability impact", height=100)
        submitted = st.form_submit_button("Ask")
    if submitted and q.strip():
        with st.spinner("Thinking..."):
            try:
                st.markdown(answer_question(q.strip()))
            except Exception as e:
                st.error(f"Something went wrong answering your question: {e}")

with tab_recs:
    c1, c2 = st.columns(2)
    with c1:
        risk = st.selectbox("Risk preference", ["low", "medium", "high"], index=0)
    with c2:
        k = st.slider("How many results?", 5, 30, 10)

    values = st.multiselect(
        "Values to emphasize",
        ["climate", "deforestation", "diversity"],
        default=["climate", "diversity"],
        help="Lower emissions/deforestation is better; higher board diversity is better.",
    )
    explain = st.checkbox("Explain why each ranked (show raw fields used)", value=True)
    refresh = st.checkbox("Force refresh this run", value=False)
    nonce = int(time.time() // 60) if refresh else 0

    if st.button("Recommend"):
        if not os.getenv("ALPHAVANTAGE_API_KEY"):
            st.warning("ALPHAVANTAGE_API_KEY is not set. Fundamentals may be unavailable and the list could be empty.")

        if not values:
            st.warning("Pick at least one value (e.g., climate).")
        else:
            with st.spinner("Scoring companies..."):
                try:
                    recs = _cached_recs(risk, tuple(values), k, explain, nonce)
                except Exception as e:
                    st.error(f"Failed to score recommendations: {e}")
                    recs = []

            if not recs:
                st.warning("No recommendations produced (rate-limit or missing keys). Try again shortly.")
            else:
                # Table
                if pd is not None:
                    df = pd.DataFrame(recs)
                    cols = [c for c in ["ticker", "name", "sector", "risk_score", "values_score", "composite"] if c in df.columns]
                    view = df[cols].copy()
                    for c in ("risk_score", "values_score", "composite"):
                        if c in view.columns:
                            view[c] = view[c].map(lambda x: f"{x:.3f}")
                    st.dataframe(view, use_container_width=True, hide_index=True)
                    csv_bytes = df.to_csv(index=False).encode("utf-8")
                else:
                    st.write(_fmt_scores(recs))
                    csv_bytes = _rows_to_csv(recs).encode("utf-8")

                # Explain (top 3)
                if explain:
                    st.markdown("### Details (top 3)")
                    for r in recs[:3]:
                        with st.expander(f"{r.get('ticker','')} — explain"):
                            d = r.get("detail", {}) or {}
                            risk_d = d.get("risk", {}) or {}
                            vals_d = d.get("values", {}) or {}
                            st.write(
                                "Risk inputs:",
                                f"beta={risk_d.get('beta')}, margin={risk_d.get('profit_margin')}, cap={risk_d.get('market_cap')}"
                            )
                            if vals_d:
                                st.write("Values inputs:")
                                for pref, info in vals_d.items():
                                    st.write(f"- {pref}: field={info.get('field')}, value={info.get('value')}")

                # CSV download
                st.download_button(
                    "Download CSV",
                    data=csv_bytes,
                    file_name="recommendations.csv",
                    mime="text/csv",
                )

                # Email draft
                st.markdown("### Email draft")
                with st.spinner("Drafting email..."):
                    try:
                        st.code(draft_email_from_recs(recs), language="markdown")
                    except Exception as e:
                        st.error(f"Failed to draft email: {e}")

# TODO: add sector filter and surfacing per-metric cut-offs